import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as  plt

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

def simulate_hypergraph(beta, gamma, H, time_step):
    '''
    beta: Transmission rate per contact
    gamma: Recovery rate
    H: Hyperedge matrice with shape [#hyperedges, #nodes]
    '''
    N = H.shape[1]
    S0 = np.ones(N)
    I0, R0 = np.zeros(N), np.zeros(N)
    index = np.random.randint(N)
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
    S, I, R = S.sum(axis = 0), I.sum(axis = 0), R.sum(axis = 0)

    return S, I, R

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