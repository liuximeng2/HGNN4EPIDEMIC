import random
import os
import numpy as np
import torch
from torch_geometric.data import Data

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

def split_dataset(data, train_ratio=0.6, valid_ratio=0.2, seed=0):
    """
    Splits the torch_geometric data object into train, validation, and test sets.
    
    Parameters:
    - data (Data): The torch_geometric data object to be split.
    - train_ratio (float): Proportion of the dataset to include in the train set.
    - valid_ratio (float): Proportion of the dataset to include in the validation set.
    
    Returns:
    - dict: A dictionary containing the train, validation, and test sets.
    """
    set_seed(seed)
    num_samples = data.sim_states.shape[0]
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * train_ratio)
    valid_size = int(num_samples * valid_ratio)
    
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]
    
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    valid_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)
    
    train_mask[train_indices] = True
    valid_mask[valid_indices] = True
    test_mask[test_indices] = True
    
    train_data = Data(
        sim_states=data.sim_states[train_mask],
        static_hgraph=data.static_hgraph,
        #static_hgraph=data.static_hgraph[train_mask],
        patient_zero=data.patient_zero[train_mask],
        train_mask=train_mask,
        val_mask=None,
        test_mask=None
    )
    valid_data = Data(
        sim_states=data.sim_states[valid_mask],
        static_hgraph=data.static_hgraph,
        #static_hgraph=data.static_hgraph[valid_mask],
        patient_zero=data.patient_zero[valid_mask],
        train_mask=None,
        val_mask=valid_mask,
        test_mask=None
    )
    test_data = Data(
        sim_states=data.sim_states[test_mask],
        static_hgraph=data.static_hgraph,
        #static_hgraph=data.static_hgraph[test_mask],
        patient_zero=data.patient_zero[test_mask],
        train_mask=None,
        val_mask=None,
        test_mask=test_mask
    )
    
    return train_data, valid_data, test_data


def init_path(directory_path):
    """
    Create a directory if it does not exist.

    Parameters:
    directory_path (str): The path of the directory to create.

    Returns:
    None
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")