import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import optim

import time
import math
import json

from model.model import MiniMindLM
from model.LMConfig import LMConfig

class Args:
    def __init__(self):
        self.device = 'cuda'
        self.batch_size = 1
        self.epoch = 6
        self.accumulation_steps = 1
        self.log_interval = 100
        self.save_interval = 100
        self.data_path = './dpo.jsonl'
        self.learning_rate = 1e-8
        self.dtype = "float16"
        self.grad_clip = 1.0


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        chosen = item['chosen']
        rejected = item['rejected']

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
        
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return [x_chosen,y_chosen,mask_chosen,x_rejected,y_rejected,mask_rejected]
        

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    
def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def dpo_loss(ref_probs, probs, beta=0.1):
    # 计算各样本平均概率
    # (batch_size, seq_len)
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 分开chosen和reject
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:] 

    # 损失为训练模型与参考模型各自"接受-拒绝概率差"的差
    # "接受-拒绝概率差"为chosen平均概率与reject平均概率的差
    # 损失含义:期望参考模型与训练模型预测概率分布相似
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)

    return loss.mean(dim=0)
    

if __name__=="__main__":
    args = Args()

    # model & ref_model(参考模型)
    lm_config = LMConfig()
    model = MiniMindLM(lm_config)
    ref_model = MiniMindLM(lm_config)
    
    # print(list(model.modules()))
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    ckp = f'./model/full_sft_512_zero.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.to(args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # data
    data = DPODataset(args.data_path,tokenizer)
    train_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        pin_memory=True
    )

    # 训练组件
    ctx = torch.cuda.amp.autocast()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epoch):
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            model.train()
            x_chosen,y_chosen,mask_chosen,x_rejected,y_rejected,mask_rejected = [item.to(args.device) for item in batch]
            
            x = torch.cat([x_chosen,x_rejected],dim=0)
            y = torch.cat([y_chosen,y_rejected],dim=0)
            mask = torch.cat([mask_chosen,mask_rejected],dim=0)

            lr = get_lr(epoch * iter_per_epoch + step, args.epoch * iter_per_epoch, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                # 获取模型预测在词表上的概率分布
                # (batch_size, seq_len, vocab_size)
                with torch.no_grad():
                    ref_logits = ref_model(x).logits
                logits = model(x).logits

                # 在其中获取标签值(y)的概率分布
                # gather用户获取对应索引值，input和index为除了dim维度外形状一致的tensor
                # (batch_size, seq_len)
                log_probs = F.log_softmax(logits, dim=2)
                probs = torch.gather(log_probs, dim=2, index=y.unsqueeze(2)).squeeze(-1)  # 获取对应索引值
                probs = probs*mask
                ref_log_probs = F.log_softmax(ref_logits, dim=2)
                ref_probs = torch.gather(ref_log_probs, dim=2, index=y.unsqueeze(2)).squeeze(-1)
                ref_probs = ref_probs*mask
                
                loss = dpo_loss(ref_probs, probs)
                loss = loss/args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} Time:{:.3f}s:'.format(
                        epoch,
                        args.epoch,
                        step,
                        iter_per_epoch,
                        loss.item() * args.accumulation_steps,
                        optimizer.param_groups[-1]['lr'],
                        spend_time))
        
        model.eval()
        torch.save(model.state_dict(),f'./checkpoint/dpo-{epoch}.pth')
            
