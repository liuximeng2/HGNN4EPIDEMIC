import torch
import numpy as np
from simulator import HyperNetSIR, set_seed

def simulate(num_sim, static_hgraph, timestep, infection_rate, recovery_rate):
        
    result = []
    target = torch.zeros(size = (num_sim, static_hgraph.shape[1]))

    for i in range(num_sim):
        print(f'# {i}th simulation')
        initial_states = torch.zeros(static_hgraph.shape[1], 3) # [S,I,R]
        index = np.nonzero(static_hgraph)[:, 1]
        index = index[torch.randint(0, index.size(0), size = (50, ) )]
        initial_states[:, 0] = 1
        initial_states[index, 0] = 0
        initial_states[index, 1] = 1

        model = HyperNetSIR(num_nodes=static_hgraph.shape[1], horizon=timestep, infection_rate=0.3, recovery_rate=0.03) # infection_rate, recover_rate, fixed_population
        preds = model(initial_states, static_hgraph, steps = None).unsqueeze(0)
        result.append(preds)
        target[i][index] = 1

    sim_states = torch.cat(result, dim = 0)
    print(sim_states.size())
    print(target.size())

    return sim_states, target

if __name__ == '__main__':

    set_seed(0)

    # Data(x=[2600, 3], y=[24], dynamic_hypergraph=[168, 500, 2600], processed_y=[168])
    data = torch.load('./data/H2ABM_small.pt')
    dynamic_hypergraph = data.dynamic_hypergraph
    static_hgraph = dynamic_hypergraph[0]
    for i in range(1, 10):
        static_hgraph = static_hgraph + dynamic_hypergraph[i]
    static_hgraph = (static_hgraph > 0) * 1
    print(f"# of total contacts: {static_hgraph.sum()}") 

    num_node = static_hgraph.shape[1]
    num_sim = 5
    timestep = 120
    infection_rate = 0.3
    recovery_rate = 0.03

    sim_states, patient_zero = simulate(num_sim, static_hgraph, timestep, infection_rate, recovery_rate)

    data.sim_states = sim_states
    data.patient_zero = patient_zero
    data.static_hgraph = static_hgraph
    data.hyperparameters = {'infection_rate': infection_rate,
                            'recovery_rate': recovery_rate,
                            'horizon': timestep}
    print(data)
    torch.save(data, './data/sim#0/Sim_H2ABM.pt')