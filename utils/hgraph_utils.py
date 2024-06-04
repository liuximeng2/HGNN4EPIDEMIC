import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HypergraphDataset(Dataset):
    def __init__(self, data):
        self.sim_states = torch.tensor(data.sim_states, dtype=torch.float)
        self.patient_zero = torch.tensor(data.patient_zero, dtype=torch.float)
        self.static_hgraph = torch.tensor(data['static_hgraph'], dtype=torch.float)
    
    def __len__(self):
        return self.sim_states.size(0)
    
    def __getitem__(self, idx):
        sample = {
            'sim_states': self.sim_states[idx],
            'patient_zero': self.patient_zero[idx],
            'static_hgraph': self.static_hgraph[idx],
        }
        return sample

def H2G(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
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