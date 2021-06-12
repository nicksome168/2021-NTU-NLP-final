import torch
import numpy as np

def handle_reproducibility(is_reproducible: bool = True) -> None:
    if is_reproducible:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def loss_fn(outputs, targets):
    pred_opt = torch.argmax(outputs, dim=1)
    return sum(pred_opt == targets)

def normalize_logits(logits):
    min_v = torch.min(logits)
    range_v = torch.max(logits) - min_v
    normalised = (logits - min_v) / range_v
    return normalised

def clean_data(data):
    for i, conv in enumerate(data):
        try:
            conv['answer'] = full2half(conv['answer'].strip())
            conv['text'] = full2half(conv['text'].strip())
            conv['question']['stem'] = full2half(conv['question']['stem'].strip())
            for idx, option in enumerate(conv['question']['choices']):
                conv['question']['choices'][idx]['text'] = full2half(option['text'].strip())
        except:
            print(f"{i}th answer has wrong format {conv['answer']}")
    return data

def full2half(str):
    res = ""
    for s in str:
        s = s.encode().decode('utf-8')
        num = ord(s)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        res += chr(num)
    return res