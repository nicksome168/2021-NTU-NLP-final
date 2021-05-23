import torch

def handle_reproducibility(is_reproducible: bool = True) -> None:
    if is_reproducible:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def loss_fn(outputs, targets):
    pred_opt = torch.argmax(outputs, dim=1)
    return sum(pred_opt == targets)