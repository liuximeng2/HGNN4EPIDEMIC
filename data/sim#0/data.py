import torch

data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen_aggH.pt')

#Connvert the list of hypergraphs to a third dimension
data.dynamic_hypergraph = torch.stack(data.dynamic_hypergraph, dim=0)
print(data)

torch.save(data, 'data/sim#0/DynamicSim_uva_ic1_pathogen_aggH.pt')