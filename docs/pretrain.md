### pretrain 一个 llm

本文参考[链接](https://github.com/jingyaogong/minimind)实现了一个混合精度训练的 pretrain 流程，model 和 tokenizer 可换成自定义的模型。

核心代码如下：
main.py

```python
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset,DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import math
import time
import gc

from model.model import Transformer
from model.LMConfig import LMConfig


class Args:
    def __init__(self):
        self.device = 'cuda'
        self.data_path = "./pretrain_data.csv"
        self.batch_size = 4
        self.num_workers = 1
        self.learning_rate = 2e-4
        self.accumulation_steps = 8
        self.dtype = "float16"
        self.epoch = 20
        self.log_interval = 100
        self.grad_clip = 1.0
        self.load = None

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        self.df = pd.read_csv(data_path).sample(frac=1.0)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))



if __name__=="__main__":
    args = Args()
    lmconfig = LMConfig()
    # model
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    if args.load is not None:
        model = torch.load(args.load)
    else:
        model = Transformer(lmconfig).to(args.device)
    print(model)
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')

    # data
    train_ds = PretrainDataset(args.data_path, tokenizer, lmconfig.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    # 训练组件
    ctx = nullcontext() if args.device == "cpu" else torch.cuda.amp.autocast()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epoch):
        start_time = time.time()
        for step,(X,Y,loss_mask) in enumerate(train_loader):
            model.train()
            X,Y,loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

            lr = get_lr(epoch * iter_per_epoch + step, args.epoch * iter_per_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            with ctx:  # 用于在训练过程中自动选择操作的精度，实现自动的混合精度训练
                out = model(X, Y)
                loss = out.last_loss / args.accumulation_steps  # 模型中自带了损失计算(cross entropy)
                loss_mask = loss_mask.view(-1)
                loss = torch.sum(loss * loss_mask) / loss_mask.sum()  # loss为一个标量值
            scaler.scale(loss).backward()  # 根据缩放因子缩放损失，避免梯度下溢

            if (step + 1) % args.accumulation_steps == 0:  # 梯度累计，累计多个batch再进行梯度更新
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，避免梯度爆炸

                scaler.step(optimizer)
                scaler.update()  # 根据当前的梯度情况动态调整缩放因子

                optimizer.zero_grad()  # 梯度清零

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
            ckp = f'./checkpoint/moe-{epoch}.pth'
            state_dict = model.state_dict()
            torch.save(state_dict, ckp)

```
