import numpy as np
import os
import random
from tqdm import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as  plt

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DiscreteDynamicHyperNetSIR(nn.Module):
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
        super(DiscreteDynamicHyperNetSIR, self).__init__()
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

        self.ff = nn.Linear(10, 20)
        self.beta = nn.Parameter(self.beta)
        self.gamma = nn.Parameter(self.gamma)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, H, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (n_nodes, one-hot encoding of states).
        H : torch.Tensor
            Dynamic incidence matrix of the hypergraph with shape (timestep, num_hyperedges, num_nodes).
        
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

        output = torch.zeros(self.num_nodes * steps * 3, dtype=torch.float, requires_grad=False).reshape(steps, self.num_nodes, 3)
        output.data[0] = x.data
        H = H.float()

        for i in tqdm(range(1, steps)):
            H_t = H[i-1]  # Get the incidence matrix for the current timestep
            new_cases = self.beta * (output[i-1, :, 0] * (H_t.T @ (self.tanh(H_t @ output[i-1, :, 1])))).unsqueeze(0)
            new_recovery = self.gamma * output.data[i-1, :, 1]

            output.data[i, :, 0] = output.data[i-1, :, 0] - new_cases
            output.data[i, :, 1] = output.data[i-1, :, 1] + new_cases - new_recovery
            output.data[i, :, 2] = output.data[i-1, :, 2] + new_recovery

            # Ensure no negative values
            output.data[i, :, :] = torch.clamp(output.data[i, :, :], min=0)

            # Normalize the state vectors so that rows sum up to the initial population
            total_population = output.data[i, :, :].sum(dim=1, keepdim=True)
            output.data[i, :, :] = output.data[i, :, :] / total_population

            # Convert probabilities to binary states based on infection and recovery chances
            for j in range(self.num_nodes):
                state_prob = output.data[i, j, :]
                if output.data[i-1, j, 0] == 1:  # Susceptible
                    if torch.rand(1).item() < (state_prob[1] * 0.8):
                        output.data[i, j, 0] = 0
                        output.data[i, j, 1] = 1
                elif output.data[i-1, j, 1] == 1:  # Infected
                    if torch.rand(1).item() < state_prob[2]:
                        output.data[i, j, 1] = 0
                        output.data[i, j, 2] = 1

                # Ensure binary state and row sum to 1
                output.data[i, j, :] = torch.nn.functional.one_hot(torch.argmax(output.data[i, j, :]), num_classes=3).float()

        return output

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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, H, steps=1):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (n_nodes, one-hot encoding of states).
        H : torch.Tensor
            Incidence matrix of the hypergraph with shape (num_hyperedges, num_nodes).
        
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

        output = torch.zeros(self.num_nodes * steps * 3, dtype=torch.float, requires_grad=False).reshape(steps, self.num_nodes, 3)
        output.data[0] = x.data
        H = H.float()

        for i in range(1, steps):
            new_cases = self.beta * (output[i-1, :, 0] * (H.T @ (H @ output[i-1, :, 1]))).unsqueeze(0)
            new_recovery = self.gamma * output.data[i-1, :, 1]

            output.data[i, :, 0] = output.data[i-1, :, 0] - new_cases
            output.data[i, :, 1] = output.data[i-1, :, 1] + new_cases - new_recovery
            output.data[i, :, 2] = output.data[i-1, :, 2] + new_recovery

            # Ensure no negative values
            output.data[i, :, :] = torch.clamp(output.data[i, :, :], min=0)

            # Normalize the state vectors so that rows sum up to the initial population
            total_population = output.data[i, :, :].sum(dim=1, keepdim=True)
            output.data[i, :, :] = output.data[i, :, :] / total_population

        return output

class DynamicHyperNetSIR(nn.Module):
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
        super(DynamicHyperNetSIR, self).__init__()
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, H, steps = None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features tensor with shape (n_nodes, one-hot encoding of states).
        H : torch.Tensor
            Dynamic incidence matrix of the hypergraph with shape (timestep, num_hyperedges, num_nodes).
        
        Returns
        -------
        torch.Tensor
            The output tensor of shape (time_step, n_nodes, probability of states),
            representing the predicted values for each node over the specified output timesteps.
        """
        if self.horizon is not None:
            steps = self.horizon

        device = x.device

        pathogen = torch.zeros(self.num_nodes * steps * 3, dtype=torch.float, requires_grad=False).reshape(steps, self.num_nodes, 3).to(device)
        pathogen.data[0] = x
        state_data = pathogen.clone().to(device)

        for i in range(1, steps):
            H_t = H[i-1].float().to(device)  # Get the incidence matrix for the current timestep
            # new_cases = self.beta * (pathogen[i-1, :, 0] * (H_t.T @ (H_t @ pathogen[i-1, :, 1]))).unsqueeze(0)
            new_cases = self.beta * (H_t.T @ (H_t @ state_data[i-1, :, 1]))
            new_recovery = self.gamma * state_data[i-1, :, 1]
            # print(f"New Cases: {new_cases.shape} || New Recovery: {new_recovery.shape}")
            # print(pathogen.data[i-1, :, 0].shape)

            pathogen.data[i, :, 0] = pathogen.data[i-1, :, 0] - new_cases
            pathogen.data[i, :, 1] = pathogen.data[i-1, :, 1] + new_cases - new_recovery
            pathogen.data[i, :, 2] = pathogen.data[i-1, :, 2] + new_recovery

            # Ensure no negative values
            pathogen.data[i, :, :] = torch.clamp(pathogen.data[i, :, :], min=0, max=1)

            # Normalize the state vectors so that rows sum up to the initial population
            # total_population = pathogen.data[i, :, :].sum(dim=1, keepdim=True)
            # pathogen.data[i, :, :] = pathogen.data[i, :, :] / total_population

            # Convert probabilities to binary states based on infection and recovery chances
            for j in range(self.num_nodes):
                pathogen_prob = pathogen.data[i, j, :]
                if state_data.data[i-1, j, 0] == 1:
                    if torch.rand(1).item() < pathogen_prob[1]:
                        state_data[i, j, 0] = 0
                        state_data[i, j, 1] = 1
                        state_data[i, j, 2] = 0
                    else:
                        state_data[i, j, 0] = 1
                        state_data[i, j, 1] = 0
                        state_data[i, j, 2] = 0
                elif state_data.data[i-1, j, 1] == 1:
                    if torch.rand(1).item() < pathogen_prob[2]:
                        if torch.rand(1).item() < 0.5:
                            state_data[i, j, 0] = 0
                            state_data[i, j, 1] = 0
                            state_data[i, j, 2] = 1
                        else:
                            state_data[i, j, 0] = 1
                            state_data[i, j, 1] = 0
                            state_data[i, j, 2] = 0
                    else:
                        state_data[i, j, 0] = 0
                        state_data[i, j, 1] = 1
                        state_data[i, j, 2] = 0
                else:
                    state_data[i, j, 0] = 0
                    state_data[i, j, 1] = 0
                    state_data[i, j, 2] = 1
                # print(f"Day {i} || Node {j} || Pathogen: {pathogen_prob} || State: {state_data.data[i, j, :]}") if j == 651 else None
                assert state_data.data[i, j, :].sum() == 1, f"Row {j} does not sum to 1: {state_data.data[i, j, :]}"
            if i == 20 or i == 40:
                print(f"Day {i} || Total Contact: {H[i-1].sum()} || Total Infected: {state_data[i, :, 1].sum()}")
        
        return pathogen, state_data
    
def plot_sir_simulation(results, window_size = 5):
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

    # Calculate the change in infected individuals
    delta_S = total_S[:-1] - total_S[1:]
    ma_delta_S = np.convolve(delta_S, np.ones(window_size)/window_size, mode='valid')

    # Generate time points
    time_points = torch.arange(results.size(0))
    delta_time_points = torch.arange(1, results.size(0))  # For delta_I
    avg_time_points = torch.arange(1 + (window_size - 1) / 2, results.size(0) - (window_size - 1) / 2)  # For moving average

    plt.figure(figsize=(12, 8))
    plt.plot(time_points, total_S, label='Susceptible', color='blue')
    plt.plot(time_points, total_I, label='Infected', color='red')
    plt.plot(time_points, total_R, label='Recovered', color='green')
    plt.plot(delta_time_points, delta_S, label='Change in Infected', color='orange', linestyle='--')
    plt.plot(avg_time_points, ma_delta_S, label=f'{window_size}-Step Moving Average', color='purple', linestyle=':')

    plt.title('Hypergraph SIR Simulation Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('simulation.png')

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

def generate_mixture_beta(num_samples, weights, alpha_params, beta_params):
    assert len(weights) == len(alpha_params) == len(beta_params), "All parameter lists must have the same length"
    assert np.isclose(sum(weights), 1), "Weights must sum to 1"
    
    mixture_samples = np.zeros(num_samples)
    
    for i, (weight, alpha, beta_param) in enumerate(zip(weights, alpha_params, beta_params)):
        num_samples_i = int(weight * num_samples)
        mixture_samples_i = np.random.beta(alpha, beta_param, num_samples_i)
        mixture_samples[:num_samples_i] = mixture_samples_i
    
    np.random.shuffle(mixture_samples)
    return mixture_samples

def generate_location_probabilities(num_edges, high_prob_edges, low_prob_edges, high_prob, low_prob):
    # location_probs = np.full(num_edges, low_prob)
    location_probs = np.random.beta(10, 5, size=num_edges)
    high_prob_indices = np.random.choice(num_edges, high_prob_edges, replace=False)
    location_probs[high_prob_indices] = high_prob
    low_prob_indices = np.random.choice(num_edges, low_prob_edges, replace=False)
    location_probs[low_prob_indices] = low_prob
    return location_probs

def generate_hypergraph(num_edges, num_nodes, high_prob_edges, low_prob_edges, high_prob, low_prob, weight, alpha, beta):
    hypergraph = np.zeros((num_edges, num_nodes))
    
    # Base probabilities for locations
    # base_probs = np.random.poisson(lam=2.0, size=num_edges)
    # base_probs = base_probs / base_probs.sum()
    # base_probs = np.random.beta(0.5, 0.5, size=num_edges)
    # base_probs = np.random.beta(2, 5, size=num_edges)
    base_probs = generate_mixture_beta(num_edges, weight, alpha, beta)
    
    for node in range(num_nodes):
        location_probs = generate_location_probabilities(num_edges, high_prob_edges, low_prob_edges, high_prob, low_prob)
        home = location_probs > 0.89
        hypergraph[:, node] = location_probs * base_probs
        hypergraph[:, node][home] = 0.95
    
    return hypergraph

def generate_temporal_hypergraph(hypergraph, num_time_steps, high_prob_edges, low_prob_edges, high_prob, low_prob, w, alpha, beta):
    num_edges, num_nodes = hypergraph.shape
    temporal_hypergraph = np.zeros((num_time_steps, num_edges, num_nodes))
    
    for t in tqdm(range(num_time_steps)):
        if t % 20 == 0:
            np.random.seed(t)
            hypergraph = generate_hypergraph(num_edges, num_nodes, high_prob_edges, low_prob_edges, high_prob, low_prob, w, alpha, beta)
        temporal_hypergraph[t] = np.random.binomial(1, hypergraph)
    
    return temporal_hypergraph