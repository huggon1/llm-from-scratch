### 自定义 lora 微调 llm

本文参考[链接](https://github.com/jingyaogong/minimind)实现了一个混合精度的 lora 微调 流程。lora 微调整体流程与 sft 类似，只需给模型中输入输出维度相同的线性层加上 lora 模块，将输出替换为原始线性层的输出和 lora 模块输出的和，同时还要调整可训练参数，单独将 lora 模块参数放入优化器中。

```
import torch
import torch.nn as nn


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
        self.data_path = './lora_identity.jsonl'
        self.learning_rate = 5e-5
        self.dtype = "float16"
        self.grad_clip = 1.0

# 自定义LoRA模块
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))

class LORADataset(Dataset):
    def __init__(self,data_path,tokenizer,max_len=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len=max_len

        self.sample = []
        with open(self.data_path,'r',encoding='utf-8') as f:
            for text in f.readlines():
                self.sample.append(json.loads(text.strip())["conversations"])

        self.bos_id = self.tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.sample)


    # 对非assistant角色的token进行mask
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
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_len)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        prompt = self.tokenizer.apply_chat_template(self.sample[index],add_generation_prompt=False)
        input_ids = prompt[:self.max_len]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

# 给输入输出维度相同的线性层加上LoRA模块
def apply_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1]).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)

def merge_lora(model):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_weight = module.lora.B.weight @ module.lora.A.weight
            base_state = module.state_dict().copy()
            # print(base_state.keys())
            base_state['weight']+= lora_weight
            base_state = {k:v for k,v in base_state.items() if 'lora' not in k}
            state_dict.update(base_state)
        else:
            state_dict.update(module.state_dict())
    # print(state_dict.keys())
    return state_dict


if __name__=="__main__":
    args = Args()

    # model
    lm_config = LMConfig()
    model = MiniMindLM(lm_config)
    apply_lora(model)  # 添加lora模块
    # print(list(model.modules()))
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    ckp = f'./model/full_sft_512_zero.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)  # LoRA 参数数量

    # data
    data = LORADataset(args.data_path,tokenizer)
    train_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        pin_memory=True
    )

    # 训练组件
    ctx = torch.cuda.amp.autocast()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))


    # 只对 LoRA 参数进行优化
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params.append(param)
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    # 训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epoch):
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            model.train()
            X,Y,loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

            lr = get_lr(epoch * iter_per_epoch + step, args.epoch * iter_per_epoch, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())

                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            if step % args.log_interval == 0:
                spend_time = time.time() - start_time
                print(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} Time:{:.3f}s:'.format(
                        epoch,
                        args.epoch,
                        step,
                        iter_per_epoch,
                        loss.item() * args.accumulation_steps,
                        optimizer.param_groups[-1]['lr'],
                        spend_time))

        if epoch>3:
            model.eval()
            save_lora(model, f'./checkpoint/lora-{epoch}.pth')


```
