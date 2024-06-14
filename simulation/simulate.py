import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from simulator import HyperNetSIR, set_seed, plot_sir_simulation

def simulate(initial_case, num_sim, dynamic_hgraph, timestep, infection_rate, recovery_rate):
    
    print(dynamic_hypergraph.shape)
    num_node = dynamic_hypergraph.shape[2]
    result = []
    hgraph = []
    target = torch.zeros(size = (num_sim, num_node))
    start = torch.randint(dynamic_hgraph.shape[0] - 20, size = (1, ))
    static_hgraph = dynamic_hypergraph[start]
    for j in range(start, start + 20):
        static_hgraph = static_hgraph + dynamic_hypergraph[j]
    static_hgraph = ((static_hgraph > 0) * 1).squeeze()


    for i in range(num_sim):

        print(f"# {i}th simulation || total contacts: {static_hgraph.sum()}") 

        initial_states = torch.zeros(static_hgraph.shape[1], 3) # [S,I,R]
        index = np.nonzero(static_hgraph)[:, 1]
        index = index[torch.randint(0, index.size(0), size = (initial_case, ) )] #patient_zero
        initial_states[:, 0] = 1
        initial_states[index, 0] = 0
        initial_states[index, 1] = 1

        model = HyperNetSIR(num_nodes=num_node, 
                            horizon=timestep, 
                            infection_rate=infection_rate, 
                            recovery_rate=recovery_rate)
        
        preds = model(initial_states, static_hgraph, steps = None).unsqueeze(0)
        #static_hgraph = static_hgraph.unsqueeze(0)
        #hgraph.append(static_hgraph)
        result.append(preds)
        target[i][index] = 1
    plot_sir_simulation(preds.squeeze())
    print('*** Simulation Completed ***')
    sim_states = torch.cat(result, dim = 0)
    #hgraph_log = torch.cat(hgraph, dim = 0)

    return sim_states, target, static_hgraph

if __name__ == '__main__':

    set_seed(0)

    # Data(x=[2600, 3], y=[24], dynamic_hypergraph=[168, 500, 2600], processed_y=[168])
    data = torch.load('./data/H2ABM_small.pt')

    #Filter out location nodes
    dynamic_hypergraph = data.dynamic_hypergraph[:,:,:2500]

    num_sim = 100
    timestep = 120
    infection_rate = 0.001
    recovery_rate = 0.03
    initial_case = 1

    sim_states, patient_zero, static_hgraph = simulate(initial_case, num_sim, dynamic_hypergraph, timestep, infection_rate, recovery_rate)

    data.sim_states = sim_states
    data.patient_zero = patient_zero
    data.static_hgraph = static_hgraph
    data.hyperparameters = {'infection_rate': infection_rate,
                            'recovery_rate': recovery_rate,
                            'horizon': timestep}
    
    delattr(data, 'dynamic_hypergraph')
    delattr(data, 'x')
    delattr(data, 'y')
    delattr(data, 'processed_y')
    print(data)
    torch.save(data, './data/sim#0/Sim_H2ABM_ic1_slow.pt')