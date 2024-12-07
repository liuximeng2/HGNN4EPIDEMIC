import torch
import os
import numpy as np
from simulator import DynamicHyperNetSIR, set_seed, plot_sir_simulation

def simulate(initial_case, num_sim, dynamic_hgraph, timestep, infection_rate, recovery_rate):
    
    # dynamic_hgraph: torch.Size([168, 500, 2500])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_timestep = dynamic_hgraph.shape[0]
    num_node = dynamic_hgraph.shape[2]
    result = []
    target = []
    pathogen = []
    model = DynamicHyperNetSIR(num_nodes=num_node, 
                    horizon=timestep, 
                    infection_rate=infection_rate, 
                    recovery_rate=recovery_rate).to(device)
    
    acc_H = []
    for i in range(num_timestep):
        H = dynamic_hgraph[i].clone()
        accu = torch.randint(2, 4, size = (1, )).item()
        if i > accu:
            for t in range(i - accu, i):
                H += dynamic_hgraph[t]  # Get the incidence matrix for the current timestep

            H = torch.where(H > 0, torch.tensor(1.0), torch.tensor(0.0))
            acc_H.append(H)
        else:
            acc_H.append(dynamic_hgraph[i])
    print(len(acc_H))
    dynamic_hgraph = acc_H

    for i in range(num_sim):
        """
        Introducing Patient Zero
        Initiating Node States
        """
        print(f'*** Simulation {i+1} ***')
        initial_states = torch.zeros(num_node, 3).to(device) # [S,I,R]
        random_day = torch.randint(0, 5, size = (1, )).item()
        non_zero = dynamic_hgraph[random_day].nonzero()[:, 1]
        indice = torch.randint(0, non_zero.size()[0], size = (initial_case, ) )
        patient_zero = non_zero[indice]
        # patient_zero = patient_zero[torch.randint(0, patient_zero.size(0), size = (initial_case, ) )] #patient_zero
        initial_states[:, 0] = 1
        initial_states[patient_zero, 0] = 0
        initial_states[patient_zero, 1] = 1
        
        pathogens, state_data = model(initial_states, dynamic_hgraph, steps = None)
        target.append(patient_zero)
        result.append(state_data)
        pathogen.append(pathogens)

    result = torch.stack(result, dim=0)
    target = torch.stack(target, dim=0)
    pathogen = torch.stack(pathogen, dim=0)
    print(target.size())
    print(result.size())
    plot_sir_simulation(pathogens)
    print('*** Simulation Completed ***')

    return result, target, pathogen, dynamic_hgraph

if __name__ == '__main__':

    set_seed(0)

    # Data(x=[2600, 3], y=[24], dynamic_hypergraph=[168, 500, 2600], processed_y=[168])
    data = torch.load('./data/h2abm/H2ABM_small.pt')

    #Filter out location nodes
    dynamic_hypergraph = data.dynamic_hypergraph[:,:,:2500]

    num_sim = 100
    timestep = 168
    infection_rate = 0.006
    recovery_rate = 0.0003
    initial_case = 1

    sim_states, patient_zero, pathogen, accH = simulate(initial_case, num_sim, dynamic_hypergraph, timestep, infection_rate, recovery_rate)

    data.dynamic_hypergraph = accH
    data.sim_states = sim_states
    data.patient_zero = patient_zero
    data.pathogen = pathogen
    data.hyperparameters = {'infection_rate': infection_rate,
                            'recovery_rate': recovery_rate,
                            'horizon': timestep}
    
    delattr(data, 'x')
    delattr(data, 'y')
    delattr(data, 'processed_y')
    print(data)
    torch.save(data, f'./data/sim#0/DynamicSim_uva_ic{initial_case}_pathogen_aggH.pt')