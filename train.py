import torch
import torch.nn as nn
from utils.hgraph_utils import *
from model.models import SDSTGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load('./data/sim#0/Sim_H2ABM.pt')
print(data)
x = data.sim_states
x = x.permute(0, 2, 3, 1)
data.sim_states = x.reshape(-1, 2600, 360)

dataset = HypergraphDataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

G = torch.tensor(H2G(data.static_hgraph), dtype = torch.float32).to(device)

model = SDSTGCN(N = 2600, 
                in_channels = 360, 
                out_channels = 256, 
                num_blocks = 2, 
                kernel_size = 3, 
                spatial_out_channels = 256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

def train(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for sample in data_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            optimizer.zero_grad()
            output = model(sim_states, G).squeeze()
            patient_zero = patient_zero.squeeze()
            loss = criterion(output, patient_zero)
            loss.backward()
            optimizer.step()
        for name, param in model.named_parameters():
            #print(f"Parameter name: {name}")
            if name == 'blocks.1.temporal2.gru.weight_hh_l1':
                print(f"Parameter value: {param}")
                print(f"Parameter shape: {param.shape}")
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

num_epochs = 50
train(model, dataloader, criterion, optimizer, num_epochs)