import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set the seed for all sources of randomness in Python, NumPy, and PyTorch.

    Args:
    seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)