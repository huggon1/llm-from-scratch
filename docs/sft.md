### sft 一个 pretrain 后的 llm

本文参考[链接](https://github.com/jingyaogong/minimind)实现了一个混合精度训练的 sft 流程，model 和 tokenizer 可换成自定义的模型。SFT（Supervised Fine-Tuning，监督微调）是一种在预训练模型基础上，使用对话数据进行微调的技术，sft 在训练时将非 assistant 的 token 的损失进行 mask，让模型专门学习 assistant 的表达方式。

jsonl 文件中的一条数据：

```
{"conversations": [{"role": "user", "content": "请用一段话描述阿里巴巴集团的企业文化。"}, {"role": "assistant", "content": "阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。公司倡导开放、透明、分享、责任的团队合作精神，鼓励员工创新、追求卓越，同时注重员工的个人成长和幸福感。阿里巴巴的企业文化还体现在其独特的“六脉神剑”价值观体系中，包括客户第一、拥抱变化、团队合作、诚信、激情、专业等六个方面，这些价值观不仅指导着公司的日常运营，也深深影响着每一位阿里人的行为准则。"}]}

```

经过 tokenizer 的 apply_chat_template()处理后

```
<s>system
你是 MiniMind，是一个有用的人工智能助手。</s>
<s>user
请用一段话描述阿里巴巴集团的企业文化。</s>
<s>assistant
阿里巴巴集团的企业文化以“客户第一、员工第二、股东第三”为核心价值观，强调“让天下没有难做的生意”的使命。公司倡导开放、透明、分享、责任的团队合作精神，鼓励员工创
新、追求卓越，同时注重员工的个人成长和幸福感。阿里巴巴的企业文化还体现在其独特的“六脉神剑”价值观体系中，包括客户第一、拥抱变化、团队合作、诚信、激情、专业等六
个方面，这些价值观不仅指导着公司的日常运营，也深深影响着每一位阿里人的行为准则。</s>
```

训练代码：

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
        self.batch_size = 4
        self.epoch = 6
        self.accumulation_steps = 8
        self.log_interval = 100
        self.save_interval = 100
        self.data_path = './sft_mini_512.jsonl'
        self.learning_rate = 5e-5
        self.dtype = "float16"
        self.grad_clip = 1.0

class SFTDataset(Dataset):
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


if __name__=="__main__":
    args = Args()
    # model
    lm_config = LMConfig()
    model = MiniMindLM(lm_config)
    tokenizer = AutoTokenizer.from_pretrained('./model/')
    ckp = f'./model/pretrain_512.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    print(model)
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    # data
    data = SFTDataset(args.data_path,tokenizer)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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

        if epoch>5:
            model.eval()
            ckp = f'./checkpoint/sft-{epoch}.pth'
            state_dict = model.state_dict()
            torch.save(state_dict, ckp)

```
