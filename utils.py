import torch

def handle_reproducibility(is_reproducible: bool = True) -> None:
    if is_reproducible:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def loss_fn(outputs, targets):
    pred_opt = torch.argmax(outputs, dim=1)
    return sum(pred_opt == targets)


def clean_data(data):
    for i, conv in enumerate(data):
        try:
            conv['answer'] = full2half(conv['answer'].strip())
        except:
            print(f"{i}th answer has wrong format {conv['answer']}")
    return data

def full2half(s):
    s = s.encode().decode('utf-8')
    num = ord(s)
    if num == 0x3000:
        num = 32
    elif 0xFF01 <= num <= 0xFF5E:
        num -= 0xfee0
    num = chr(num)
    return num