import torch

data = torch.load('./data/h2abm/H2ABM_small.pt')
dynamic_hypergraph = data.dynamic_hypergraph[:,:,:2500]

# Check the distribution of the each nodes total interaction
total_interactions = torch.sum(dynamic_hypergraph, dim=0)
total_interactions = torch.sum(total_interactions, dim=0).float()
# Plot the total number of interactions for each node using abr plot
import matplotlib.pyplot as plt
plt.bar(range(total_interactions.size(0)), total_interactions)
plt.ylim(0, 1000)
plt.savefig('total_interactions.png')
# print indices with interaction greater than 500
print(torch.nonzero(total_interactions > 500).squeeze())