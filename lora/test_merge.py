import torch
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from transformers import AutoTokenizer
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
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

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}

            module.lora.load_state_dict(lora_state)

def merge_lora(model):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_weight = module.weight + module.lora.B.weight @ module.lora.A.weight
            base_state = module.state_dict().copy()
            state_dict.update({f"{name}.weight": lora_weight})
        else:
            state_dict.update(module.state_dict())
    # print(state_dict.keys())
    return state_dict

def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora

lm_config = LMConfig()

model = MiniMindLM(lm_config)

s1 = model.state_dict()
apply_lora(model)
s2 = merge_lora(model)
print(s1.keys())
print("===")
print("===")
print("===")
print("===")

print(s2.keys())



tokenizer = AutoTokenizer.from_pretrained('./model/')
test_text = "This is a test sentence."
inputs = tokenizer(test_text, return_tensors='pt')

model = MiniMindLM(lm_config)
model.load_state_dict(torch.load("/code/demo/lora/model/full_sft_512_zero.pth"))
output0 = model(**inputs)

model = MiniMindLM(lm_config)
model.load_state_dict(torch.load("/code/demo/lora/model/full_sft_512_zero.pth"))
load_lora(model,"/code/demo/lora/checkpoint/lora-4.pth")
output1 = model(**inputs)

model = MiniMindLM(lm_config)
model.load_state_dict(torch.load("/code/demo/lora/checkpoint/merge-4.pth"))
output2 = model(**inputs)


print(output0)
print(output1)
print(output2)

outputs_are_equal = torch.allclose(output0, output1, atol=1e-6)
print(f"Outputs are equal01: {outputs_are_equal}")

outputs_are_equal = torch.allclose(output0, output2, atol=1e-6)
print(f"Outputs are equal02: {outputs_are_equal}")

outputs_are_equal = torch.allclose(output1, output2, atol=1e-6)
print(f"Outputs are equal12: {outputs_are_equal}")
