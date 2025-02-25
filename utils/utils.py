import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from utils.hgraph_utils import *

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

def split_dataset(data, train_ratio=0.6, valid_ratio=0.2, seed=0, is_multiple_instance=False):
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
    print(f"Train size: {train_size}, Validation size: {valid_size}, Test size: {num_samples - train_size - valid_size}")
    
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
        dynamic_hgraph=data.dynamic_hypergraph,
        patient_zero=data.patient_zero[train_mask],
        forecast_label=data.forecast_label[train_mask],
        dynamic_edge_list = [row for row, mask in zip(data.dynamic_edge_list, train_mask) if mask] if is_multiple_instance else data.dynamic_edge_list,
        # dynamic_edge_list=data.dynamic_edge_list[list(train_mask)] if is_multiple_instance else data.dynamic_edge_list,
        train_mask=train_mask,
        val_mask=None,
        test_mask=None
    )
    valid_data = Data(
        sim_states=data.sim_states[valid_mask],
        dynamic_hgraph=data.dynamic_hypergraph,
        patient_zero=data.patient_zero[valid_mask],
        forecast_label=data.forecast_label[valid_mask],
        dynamic_edge_list = [row for row, mask in zip(data.dynamic_edge_list, valid_mask) if mask] if is_multiple_instance else data.dynamic_edge_list,
        val_mask=valid_mask,
        test_mask=None
    )
    test_data = Data(
        sim_states=data.sim_states[test_mask],
        dynamic_hgraph=data.dynamic_hypergraph,
        patient_zero=data.patient_zero[test_mask],
        forecast_label=data.forecast_label[test_mask],        
        dynamic_edge_list = [row for row, mask in zip(data.dynamic_edge_list, test_mask) if mask] if is_multiple_instance else data.dynamic_edge_list,
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

def load_data(args, interested_interval):   

    print("Processing data for forecasting") if args.forecast else print("Processing data for source detection")

    if args.dataset == 'UVA':
        data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen.pt')
        data.forecast_label = data.sim_states[:, interested_interval:interested_interval + args.num_for_predict, :, 1].detach()
        data.dynamic_hypergraph = data.dynamic_hypergraph[0:interested_interval, :, :]
        data.dynamic_edge_list = process_hyperedges_incidence(data.dynamic_hypergraph, interested_interval)
        print(f"forecast label: {data.forecast_label.shape}")
        # data.forecast_label = retain_first_one(forecast_label)
        print(f"forecast label: {data.forecast_label.shape}")
        horizon = data.hyperparameters['horizon']
        assert data is not None, 'Data not found'
        args.num_hyperedge = 500
        args.num_of_vertices = 2500
    
    if args.dataset == 'EpiSim':
        data = torch.load("data/epiSim/simulated_epi.pt")
        data.sim_states = data.sim_states.float()
        data.patient_zero = torch.unsqueeze(data.patient_zero, 1).to(dtype=torch.int64)
        data.dynamic_hypergraph = data.dynamic_hypergraph.float()
        horizon = data.sim_states.shape[1]
        # pre_symptom = data.sim_states[:, interested_interval:interested_interval + args.num_for_predict, :, 1]
        symptom = data.sim_states[:, interested_interval:interested_interval + args.num_for_predict, :, 2]
        critical = data.sim_states[:, interested_interval:interested_interval + args.num_for_predict, :, 3]
        #combine the three states into one using OR operation
        data.forecast_label = torch.logical_or(symptom, critical) #[num_sampel, num_individual, timestep]
        # data.forecast_label = torch.logical_or(torch.logical_or(pre_symptom, symptom), critical) #[num_sampel, num_individual, timestep]
        #print(f"forecast label shape: {data.forecast_label.shape}")
        #for all rows of num_individual, only save the first entry with 1 and set the rest to 0. FOR EXAMPLE, [[0, 1, 1, 1, 0], [1, 0, 0, 0, 0]] -> [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]
        # data.forecast_label = retain_first_one(data.forecast_label)
        #print(f"forecast label shape: {data.forecast_label.shape}")

        data.dynamic_hypergraph = data.dynamic_hypergraph[:, 0:interested_interval, :, :]
        data.dynamic_hypergraph = data.dynamic_hypergraph.permute(0, 1, 3, 2)
        data.dynamic_edge_list = process_hyperedges_incidence(data.dynamic_hypergraph, interested_interval, multiple_instance=True)
        args.in_channels = 10
        args.num_of_vertices = 10000
        args.num_hyperedge = 11


    assert horizon > args.timestep_hidden, 'Horizon should be greater than timestep_hidden'

    data.sim_states[:, 0:args.timestep_hidden, :, :] = 0 # mask the first timestep_hidden timesteps
    data.sim_states = data.sim_states[:, 0:interested_interval, :, :] #[num_sample, time, num_nodes, num_features]
    args.len_input = interested_interval
    if not args.forecast:
        args.num_for_predict = 1 # for source detection

    return data, args

def create_dataloader(args, data, seed):

    if args.dataset == 'UVA':
        train_data, valid_data, test_data = split_dataset(data, seed=seed)
        train_data, valid_data, test_data = DynamicHypergraphDataset(train_data), DynamicHypergraphDataset(valid_data), DynamicHypergraphDataset(test_data)
    if args.dataset == 'EpiSim':
        train_data, valid_data, test_data = split_dataset(data, seed=seed, is_multiple_instance=True)
        train_data, valid_data, test_data = DynamicHypergraphDataset(train_data, multiple_instance=True), DynamicHypergraphDataset(valid_data, multiple_instance=True), DynamicHypergraphDataset(test_data, multiple_instance=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def retain_first_one(binary_tensor):
    """
    For each row in the binary tensor, retain only the first occurrence of 1 in the last dimension,
    setting all subsequent ones to 0.
    
    Parameters:
    binary_tensor (numpy.ndarray): A 3D binary tensor of shape [num_sample, num_individual, timestep].

    Returns:
    numpy.ndarray: The transformed tensor with only the first occurrence of 1 retained per row.
    """
    binary_tensor = binary_tensor.permute(0, 2, 1)
    transformed_tensor = np.zeros_like(binary_tensor)  # Initialize tensor with zeros

    # Iterate over samples and individuals
    for sample_idx in range(binary_tensor.shape[0]):
        for individual_idx in range(binary_tensor.shape[1]):
            first_one_idx = np.argmax(binary_tensor[sample_idx, individual_idx])  # Find first occurrence of 1
            if binary_tensor[sample_idx, individual_idx, first_one_idx] == 1:
                transformed_tensor[sample_idx, individual_idx, first_one_idx] = 1  # Retain only the first 1
    transformed_tensor =torch.tensor(transformed_tensor).permute(0, 2, 1)
    return transformed_tensor[:, 1:, :]

def retain_first_one_2d(binary_tensor):
    """
    For each row in the binary tensor, retain only the first occurrence of 1 in the last dimension,
    setting all subsequent ones to 0.
    
    Parameters:
    binary_tensor (numpy.ndarray): A 2D binary tensor of shape [timestep, num_individual].

    Returns:
    numpy.ndarray: The transformed tensor with only the first occurrence of 1 retained per row.
    """
    transformed_tensor = np.zeros_like(binary_tensor)  # Initialize tensor with zeros

    # Iterate over samples and individuals
    for individual_idx in range(binary_tensor.shape[1]):
        first_one_idx = np.argmax(binary_tensor[:, individual_idx])  # Find first occurrence of 1
        if binary_tensor[first_one_idx, individual_idx] == 1:
            transformed_tensor[first_one_idx, individual_idx] = 1  # Retain only the first 1
    return transformed_tensor