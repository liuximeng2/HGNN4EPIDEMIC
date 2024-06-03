import numpy as np
import random
from scipy.integrate import odeint
import matplotlib.pyplot as  plt

import torch
import torch.nn as nn

class HyperNetSIR(nn.Module):
    """
    Network-based SIR (Susceptible-Infected-Recovered)

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes in the graph representing individuals or groups. Default: None.
    horizon : int, optional
        Number of future time steps to simulate. If None, a single step is simulated unless overridden in the forward method.
    infection_rate : float, optional
        Initial infection rate parameter, representing the rate at which susceptible individuals become infected. Default: 0.01.
    recovery_rate : float, optional
        Initial recovery rate parameter, representing the rate at which infected individuals recover. Default: 0.038.
    population : int, optional
        Total population considered in the model. If None, the sum of the initial conditions (susceptible, infected, recovered) is used as the total population.

    Returns
    -------
    torch.Tensor
        A tensor of shape (time_step, num_nodes, 3), representing the predicted number of susceptible, infected, and recovered individuals at each timestep for each node.
        Each row corresponds to a timestep, with the columns representing the susceptible, infected, and recovered counts respectively for each node.
    """
    def __init__(self, num_nodes=None, horizon=None, infection_rate=0.01, recovery_rate=0.038, population=None):
        super(HyperNetSIR, self).__init__()
        self.pop = population
        self.horizon = horizon
        self.num_nodes = num_nodes

        self.beta = torch.abs(torch.rand(num_nodes))
        self.gamma = torch.abs(torch.rand(num_nodes))

        if infection_rate is not None:
            new_weights = torch.zeros_like(self.beta.data) + torch.FloatTensor([infection_rate])
            self.beta.data = new_weights

        if recovery_rate is not None:
            new_weights = torch.zeros_like(self.gamma.data) + torch.FloatTensor([recovery_rate])
            self.gamma.data = new_weights

        self.beta = nn.Parameter(self.beta)
        self.gamma = nn.Parameter(self.gamma)

    def forward(self, x, adj, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (n_nodes, one-hot encoding of states).
        adj : torch.Tensor
            Static adjacency matrix of the graph with shape (num_nodes, num_nodes).
        states : torch.Tensor, optional
            States of the nodes if available, with the same shape as x. Default: None.
        dynamic_adj : torch.Tensor, optional
            Dynamic adjacency matrix if available, with shape similar to adj but possibly varying over time. Default: None.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (time_step, n_nodes, probability of states),
            representing the predicted values for each node over the specified output timesteps.
        """
        if self.pop is not None:
            pop = self.pop
        else:
            pop = x.sum()
        if self.horizon is not None:
            steps = self.horizon

        # rescale = nn.Softmax(dim=0)

        output = torch.zeros(self.num_nodes*steps*3, dtype=torch.float, requires_grad = False).reshape(steps, self.num_nodes, 3)
        output.data[0] = x.data
        adj = adj.float()
        adj_hyper = adj.T @ adj
        min_val = adj_hyper.min()
        max_val = adj_hyper.max()
        normalized_tensor = (adj_hyper - min_val) / (max_val - min_val)

        for i in range(1, steps):
            #new_cases = self.beta*(output.data[i-1,:,0]*(adj @ output.data[i-1,:,1])).unsqueeze(0)
            #new_cases = self.beta*(output.data[i-1,:,0]*(adj.T @ (adj @ output.data[i-1,:,1]))).unsqueeze(0)
            #print((adj.T @ (adj @ output.data[i-1,:,1])).size())
            new_cases = self.beta * (output[i-1, :, 0] * (normalized_tensor @ output[i-1, :, 1])).unsqueeze(0)

            new_recovery = self.gamma*output.data[i-1,:,1]

            output.data[i,:,0] = output.data[i-1,:,0] - new_cases
            output.data[i,:,1] = output.data[i-1,:,1] + new_cases - new_recovery
            output.data[i,:,2] = output.data[i-1,:,2] + new_recovery

            # total = output.data[i, :, 0] + output.data[i, :, 1] + output.data[i, :, 2]
            # print(total) if i == 2 else None
            # output.data[i, :, 0] = output.data[i, :, 0] / total
            # output.data[i, :, 1] = output.data[i, :, 1] / total
            # output.data[i, :, 2] = output.data[i, :, 2] / total


            # output.data[i,:] = rescale(output.data[i,:])

        return output
    
def plot_sir_simulation(results):
    """
    Plot the SIR simulation results over time.

    Parameters
    ----------
    results : torch.Tensor
        Output tensor from the simulate_hyper_sir function, shape (steps, num_nodes, 3).
    """
    # Calculate the sum over all nodes for S, I, R at each time step
    total_S = results[:, :, 0].sum(axis=1)
    total_I = results[:, :, 1].sum(axis=1)
    total_R = results[:, :, 2].sum(axis=1)

    # Generate time points
    time_points = torch.arange(results.size(0))

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, total_S, label='Susceptible', color='blue')
    plt.plot(time_points, total_I, label='Infected', color='red')
    plt.plot(time_points, total_R, label='Recovered', color='green')

    plt.title('Hypergraph SIR Simulation Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('simulation.png')

# Outdated due to intergral calculation
def simulate_graph(beta, gamma, A, time_step):
    '''
    beta: Transmission rate per contact
    gamma: Recovery rate
    H: Adjacency matrix
    '''
    N = A.shape[0]

    S0 = np.ones(N)
    S0[0] = 0
    I0 = np.zeros(N)
    I0[0] = 1
    R0 = np.zeros(N)
    y0 = np.concatenate([S0, I0, R0])

    t = np.linspace(0, time_step, time_step)

    def deriv(y, t, N, beta, gamma, A):
        S, I, R = y.reshape(3, N)

        new_infections = beta * S * np.dot(A, I)
        new_recoveries = gamma * I
        dSdt = -new_infections
        dIdt = new_infections - new_recoveries
        dRdt = new_recoveries

        return np.concatenate([dSdt, dIdt, dRdt])

    ret = odeint(deriv, y0, t, args=(N, beta, gamma, A))
    S, I, R = ret.T.reshape(3, N, len(t))
    S, I, R = S.sum(axis = 0), I.sum(axis = 0), R.sum(axis = 0)

    return S, I, R

# Outdated due to intergral calculation
def simulate_hypergraph(beta, gamma, H, time_step):
    '''
    beta: Transmission rate per contact
    gamma: Recovery rate
    H: Hyperedge matrice with shape [#hyperedges, #nodes]
    '''
    H = np.array(H) if isinstance(H, torch.Tensor) else H
    N = H.shape[1]
    M = H.shape[0]
    S0 = np.ones(N)
    I0, R0 = np.zeros(N), np.zeros(N)
    #index = np.random.randint(N, size = (10))
    index = np.nonzero(H)[1][0:10]
    S0[index] = 0
    I0[index] = 1

    y0 = np.concatenate([S0, I0, R0])
    t = np.linspace(0, time_step, time_step)

    def deriv(y, t, N, M, beta, gamma, H):
        S, I, R = y.reshape(3, N)
        infection_contributions = np.dot(H.T, np.dot(H, I)) #Aggregation
        new_infections = beta * S * infection_contributions
        new_recoveries = gamma * I
        dSdt = -new_infections
        dIdt = new_infections - new_recoveries
        dRdt = new_recoveries
        return np.concatenate([dSdt, dIdt, dRdt])

    ret = odeint(deriv, y0, t, args=(N, M, beta, gamma, H))
    S, I, R = ret.T.reshape(3, N, len(t))
    print(S.shape)
    S, I, R = S.sum(axis = 0), I.sum(axis = 0), R.sum(axis = 0)

    return S, I, R


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

if __name__ == '__main__':
    beta = 0.0003 
    gamma = 0.1
    time_step = 200
    N = 100
    M = 20
    
    # Generate a random hyperedge matrix
    H = np.random.randint(0, 2, (M, N))
    S, I, R = simulate_hypergraph(beta, gamma, H, time_step)

    time_step = np.linspace(0, time_step, time_step)
    plt.figure(figsize=(10, 6))
    plt.plot(time_step, I, label='Infected')
    plt.plot(time_step, S, label='Susceptible')
    plt.plot(time_step, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('HyperGraph SIR Model Simulation for Individuals')
    plt.legend()
    plt.show()