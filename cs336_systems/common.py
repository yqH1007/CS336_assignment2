import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def synchronize(device: str):
    if device=="cuda":
        torch.cuda.synchronize()
    elif device=="mps":
        torch.mps.synchronize()
        