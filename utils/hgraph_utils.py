import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HypergraphDataset(Dataset):
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

def H2G(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H for HGNN
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H, dtype = float).T
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2.setflags(write=True)
    DV2[np.isinf(DV2)] = 0
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def Hyper2Graph(static_hgraph):
    """
    Convert a single hypergraph simulation to a graph.

    Parameters:
    static_hgraph (torch.Tensor): The static hypergraph tensor of shape (num_hedges, num_nodes).
    sim_states (torch.Tensor): The simulation states tensor of shape (num_timesteps, num_nodes).

    Returns:
    tuple: A tuple containing:
        - node_features (torch.Tensor): Node features tensor of shape (num_timesteps, num_nodes).
        - labels (torch.Tensor): Labels tensor of shape (num_timesteps).
        - edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
        - edge_weight (torch.Tensor): Edge weight tensor of shape (num_edges).
    """
    num_hedges, num_nodes = static_hgraph.size()

    # Initialize the adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    # For each hyperedge, add edges to the adjacency matrix
    for hedge in range(num_hedges):
        nodes = static_hgraph[hedge].nonzero().squeeze()
        if nodes.dim() > 0:
            for u in nodes:
                for v in nodes:
                    if u != v:
                        adj_matrix[u, v] = 1.0

    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()

    return edge_index

def H2D(hypergraph):
    """
    Converts a hypergraph incidence matrix to a dictionary format.
    
    Args:
        hypergraph (torch.Tensor): Incidence matrix of shape [#nodes, #edges].
        
    Returns:
        hypergraph_dict (dict): Dictionary where keys are hyperedges and values are lists of nodes.
    """
    hypergraph_dict = {}
    num_nodes, num_edges = hypergraph.shape
    
    for edge in range(num_edges):
        nodes_in_edge = torch.where(hypergraph[:, edge] == 1)[0].tolist()
        hypergraph_dict[edge] = nodes_in_edge
    
    return hypergraph_dict

def H2EL(hypergraph):

    edge_idx = 2500
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int)
    edge_index = torch.LongTensor(edge_index)
    
    return edge_index

def ExtractV2E(edge_index):
    # Assume edge_index = [V|E;E|V]
    # First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = 2500
    num_hyperedges = 500
    if not ((num_nodes+num_hyperedges-1) == edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return edge_index