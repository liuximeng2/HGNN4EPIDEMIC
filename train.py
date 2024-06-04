import os
import torch
import torch.nn as nn
from utils.hgraph_utils import *
from utils.utils import *
from model.models import SDSTGCN

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            static_hgraph = sample['static_hgraph']
            G = torch.zeros(size = (static_hgraph.shape[0], static_hgraph.shape[2], static_hgraph.shape[2])).to(device)
            for g in range(G.shape[0]):
                G[g] = torch.tensor(H2G(static_hgraph[g]), dtype = torch.float32)
            optimizer.zero_grad()
            output = model(sim_states, G).squeeze()
            patient_zero = patient_zero.squeeze() * 10
            loss = criterion(output, patient_zero)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {(loss.item()):.4f}')

        if epoch % 5 == 0:
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, train_loader, device)
            print(f'Train: MRR: {(mrr * 100):.2f}, Hits@1: {(hits_at_1 * 100):.2f}, Hits@10: {(hits_at_10 * 100):.2f}, Hits@100: {(hits_at_100 * 100):.2f}')
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, valid_loader, device)
            print(f'Valid: MRR: {(mrr * 100):.2f}, Hits@1: {(hits_at_1 * 100):.2f}, Hits@10: {(hits_at_10 * 100):.2f}, Hits@100: {(hits_at_100 * 100):.2f}')
        

def evaluate_model(model, data_loader, device):
    model.eval()
    mrr = 0
    hits_at_1 = 0
    hits_at_10 = 0
    hits_at_100 = 0
    total_samples = 0

    with torch.no_grad():
        for sample in data_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)

            static_hgraph = sample['static_hgraph']
            G = torch.zeros(size = (static_hgraph.shape[0], static_hgraph.shape[2], static_hgraph.shape[2])).to(device)
            for g in range(G.shape[0]):
                G[g] = torch.tensor(H2G(static_hgraph[g]), dtype = torch.float32)
            
            output = model(sim_states, G).squeeze(dim=-1)

            # Ensure the output and target have matching dimensions
            batch_size, num_nodes = output.size()
            target = patient_zero.view(batch_size, num_nodes)

            for i in range(batch_size):
                target_indices = torch.nonzero(target[i]).view(-1)
                output_sorted, indices = torch.sort(output[i], descending=True)
                ranks = torch.where(indices.unsqueeze(1) == target_indices)[0].float() + 1

                mrr += torch.sum(1.0 / ranks).item()
                hits_at_1 += torch.sum(ranks <= 1).item()
                hits_at_10 += torch.sum(ranks <= 10).item()
                hits_at_100 += torch.sum(ranks <= 100).item()
                total_samples += len(target_indices)

    mrr /= total_samples
    hits_at_1 /= total_samples
    hits_at_10 /= total_samples
    hits_at_100 /= total_samples

    return mrr, hits_at_1, hits_at_10, hits_at_100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = torch.load('./data/sim#0/Sim_H2ABM.pt')
print(data)

timestep_hidden = 0

x = data.sim_states
x = x[:, timestep_hidden:, :, :]
x = x.permute(0, 2, 3, 1)
print(x.size())
data.sim_states = x.reshape(-1, 2500, 360)
print(data.sim_states[0, 0, :])

train_data, valid_data, test_data = split_dataset(data)

train_data, valid_data, test_data = HypergraphDataset(train_data), HypergraphDataset(valid_data), HypergraphDataset(test_data)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

model = SDSTGCN(N = 2500, 
                in_channels = 360, 
                out_channels = 256, 
                num_blocks = 2, 
                kernel_size = 3, 
                spatial_out_channels = 256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

num_epochs = 50
train(model, train_loader, valid_loader, criterion, optimizer, num_epochs)