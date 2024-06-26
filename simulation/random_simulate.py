import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
from simulator import HyperNetSIR, set_seed, plot_sir_simulation

def simulate(num_node, num_hedge, edge_prob, initial_case, num_sim, timestep, infection_rate, recovery_rate):
    
    result = []
    hgraph = []
    target = torch.zeros(size = (num_sim, num_node))

    edge_index = erdos_renyi_graph(num_node, edge_prob)
    adj= torch.zeros((num_node, num_node), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1.0
    random_hgraph = adj[:num_hedge, :]

    for i in range(num_sim):

        print(f"# {i}th simulation || total contacts: {random_hgraph.sum()}") 

        initial_states = torch.zeros(num_node, 3) # [S,I,R]
        index = np.nonzero(random_hgraph)[:, 1]
        index = index[torch.randint(0, index.size(0), size = (initial_case, ) )] #patient_zero
        initial_states[:, 0] = 1
        initial_states[index, 0] = 0
        initial_states[index, 1] = 1

        model = HyperNetSIR(num_nodes=num_node, 
                            horizon=timestep, 
                            infection_rate=infection_rate, 
                            recovery_rate=recovery_rate)
        
        preds = model(initial_states, random_hgraph, steps = None).unsqueeze(0)
        #random_hgraph = random_hgraph.unsqueeze(0)
        #hgraph.append(random_hgraph)
        result.append(preds)
        target[i][index] = 1

    print(preds.squeeze()[5, 0:10, :])

    plot_sir_simulation(preds.squeeze())
    print('*** Simulation Completed ***')
    sim_states = torch.cat(result, dim = 0)
    #hgraph_log = torch.cat(hgraph, dim = 0)

    return sim_states, target, random_hgraph

if __name__ == '__main__':

    set_seed(0)
    num_sim = 100
    num_node = 10000
    num_hedge = 2000
    edge_prob = 0.005
    timestep = 120
    infection_rate = 0.003
    recovery_rate = 0.01
    initial_case = 1

    sim_states, patient_zero, static_hgraph = simulate(num_node, num_hedge, edge_prob,initial_case, num_sim, timestep, infection_rate, recovery_rate)

    data = Data(static_hgraph=static_hgraph)

    data.sim_states = sim_states
    data.patient_zero = patient_zero
    data.hyperparameters = {'infection_rate': infection_rate,
                            'recovery_rate': recovery_rate,
                            'horizon': timestep,
                            'initial_cases': initial_case}
    
    print(data)
    torch.save(data, f'./data/sim#0/n1000_random_ic{initial_case}.pt')