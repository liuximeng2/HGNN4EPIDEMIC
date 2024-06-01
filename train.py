import torch
from utils.hgraph_utils import *
from model.models import HGNN

data = torch.load('./data/sim#0/Sim_H2ABM_small.pt')
print(data)
x = data.x
G = torch.tensor(H2G(data.static_hgraph), dtype = torch.float32)

spatial_model = HGNN(in_ch=3, n_class=1, n_hid=256, dropout=0.3)
output = spatial_model(x, G)

print(output.size())