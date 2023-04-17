import torch

def get_device():
    """
    The function returns "cuda" if a GPU is available, otherwise it returns "cpu".
    :return: The function `get_device()` returns the string "cuda" if a CUDA-enabled GPU is available,
    otherwise it returns the string "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"