import torch
import numpy as np
import matplotlib.pyplot as plt
from simulation.simulator import HyperNetSIR, plot_sir_simulation

# Data(x=[2600, 3], y=[24], dynamic_hypergraph=[168, 500, 2600], processed_y=[168])
data = torch.load('./data/H2ABM_small.pt')
dynamic_hypergraph = data.dynamic_hypergraph
static_hgraph = dynamic_hypergraph[0]
for i in range(1, 10):
    static_hgraph = static_hgraph + dynamic_hypergraph[i]
static_hgraph = (static_hgraph > 0) * 1
#static_hgraph = torch.randint(0, 2, (50, 200))
print(f"# of total contacts: {static_hgraph.sum()}") 

initial_states = torch.zeros(static_hgraph.shape[1],3) # [S,I,R]
index = np.nonzero(static_hgraph)[:, 1][:100]
print(index.size())
initial_states[:, 0] = 1
# set infected individual: 10
initial_states[index, 0] = 0
initial_states[index, 1] = 1

model = HyperNetSIR(num_nodes=static_hgraph.shape[1], horizon=120, infection_rate=0.3, recovery_rate=0.03) # infection_rate, recover_rate, fixed_population
preds = model(initial_states, static_hgraph, steps = None)
plot_sir_simulation(preds)


data.sim_states = preds
data.patient_zero = index
data.static_hgraph = static_hgraph
data.hyperparameters = {'infection_rate': 0.3,
                        'recovery_rate': 0.03,
                        'horizon': 120}
print(data)
torch.save(data, './data/Sim_H2ABM_small.pt')