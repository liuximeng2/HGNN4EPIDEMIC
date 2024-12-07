import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import os

class StaticHypergraphDataset(Dataset):
    def __init__(self, data):
        self.sim_states = data.sim_states.clone().detach()
        self.patient_zero = data.patient_zero.clone().detach()
        self.static_hgraph = data.static_hgraph.clone().detach()
    
    def __len__(self):
        return self.sim_states.size(0)
    
    def __getitem__(self, idx):
        sample = {
            'sim_states': self.sim_states[idx],
            'patient_zero': self.patient_zero[idx],
            'static_hgraph': self.static_hgraph,
        }
        return sample
    

class DynamicHypergraphDataset(Dataset):
    def __init__(self, data):
        self.sim_states = data.sim_states.clone().detach()
        self.patient_zero = data.patient_zero.clone().detach()
        self.dynamic_hgraph = data.dynamic_hgraph.clone().detach()
        self.forecast_label = data.forecast_label.clone().detach()
        self.dynamic_edge_list = data.dynamic_edge_list
    
    def __len__(self):
        return self.sim_states.size(0)
    
    def __getitem__(self, idx):
        sample = {
            'sim_states': self.sim_states[idx],
            'patient_zero': self.patient_zero[idx],
            'forecast_label': self.forecast_label[idx],
            'dynamic_hgraph': self.dynamic_hgraph,
            'dynamic_edge_list': self.dynamic_edge_list
        }
        return sample
    
def process_hyperedges_incidence(H, horizon):
        """
        Processes the incidence matrix to create edge_index for HypergraphConv

        Args:
            H (torch.Tensor): Incidence matrix of shape [num_hyperedge, num_node]

        Returns:
            torch.Tensor: edge_index of shape [2, num_edges]
        """
        dynamic_edge_list = []
        for t in range(horizon):
            # Find indices where the incidence matrix is non-zero
            hyperedge_indices, node_indices = H[t].nonzero(as_tuple=True)

            # edge_index: [2, num_edges] where the first row is node indices, and the second row is hyperedge indices
            edge_index = torch.stack([node_indices, hyperedge_indices], dim=0)
            dynamic_edge_list.append(edge_index)
        return dynamic_edge_list

def find_influenced_nodes(initial_nodes, hypergraph_sequence):
    # Convert initial nodes to a set for efficient membership testing
    influenced_nodes = set(initial_nodes.tolist())
    # Iterate over the hypergraph sequence in reverse order
    for hypergraph in reversed(hypergraph_sequence):
        # Initialize a new set to store additional influenced nodes from this hypergraph
        # Check each hyperedge for intersections with influenced nodes
        for edge_idx in range(hypergraph.shape[0]):
            edge_nodes = np.nonzero(hypergraph[edge_idx])  # Nodes in the current hyperedge
            # If any node in edge_nodes is in the influenced set, add all edge nodes to new_influences
            edge_nodes = set(edge_nodes.flatten().tolist())
            if influenced_nodes.intersection(edge_nodes):
                influenced_nodes.update(edge_nodes)
    
    return influenced_nodes

def get_subgraph_nodes(sim_states, dynamic_edge_list, timestep_hidden):

    sim_state_at_hidden = sim_states[timestep_hidden, :, :]
    edge_list_b4_hidden = dynamic_edge_list[:timestep_hidden]

    infected_nodes = sim_state_at_hidden[:, 1].nonzero(as_tuple=True)[0]
    recovered_nodes = sim_state_at_hidden[:, 2].nonzero(as_tuple=True)[0]

    infected_nodes = torch.cat((infected_nodes, recovered_nodes))
    influenced_nodes = find_influenced_nodes(infected_nodes, edge_list_b4_hidden)
    influenced_nodes = torch.tensor(list(influenced_nodes))

    return influenced_nodes

def get_subgraph_edges(edges, subgraph_nodes):
    # Convert subgraph nodes to a set for quick lookup
    subgraph_node_set = set(subgraph_nodes.tolist())
    # Initialize a list to hold edges in the subgraph
    list_edges = []
    
    for edge in edges:
        subgraph_edges = []
        edge = edge[0]
    # Iterate over each edge
        for i in range(edge.shape[1]):
            node1, node2 = edge[:, i]  # Get the two nodes in the current edge
            # Check if both nodes are in the subgraph node set
            if node1.item() in subgraph_node_set and node2.item() in subgraph_node_set:
                # If both nodes are in the subgraph, add the edge to subgraph_edges
                subgraph_edges.append([node1, node2])
        subgraph_edges = torch.tensor(subgraph_edges).T
        list_edges.append(subgraph_edges)

    return list_edges

def reindex_edges(edges, node_indices):
    """
    Reindex edges based on the given node indices.
    
    Parameters:
    - edges: torch.Tensor of shape [2, num_edges], where each column represents an edge.
    - node_indices: torch.Tensor containing the nodes in the subgraph.
    
    Returns:
    - reindexed_edges: torch.Tensor of shape [2, num_edges] with nodes re-indexed to 0:num_nodes-1.
    """
    result = []
    # Create a mapping from original indices to re-indexed (0:num_nodes-1) indices
    reindex_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
    # Reindex the edges using the mapping
    for edge in edges:
        reindexed_edges = edge.clone()
        for i in range(edge.shape[1]):
            reindexed_edges[0, i] = reindex_map[edge[0, i].item()]
            reindexed_edges[1, i] = reindex_map[edge[1, i].item()]
        result.append(reindexed_edges)
    return result