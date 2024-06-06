import logging
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from utils.hgraph_utils import *
from utils.utils import *
from model.models import TMLP, THGNN

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, logging, device):
    for epoch in tqdm(range(num_epochs + 1)):
        model.train()
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            sim_states = sim_states.contiguous().view(-1, sim_states.size(2),  sim_states.size(3)).to(device)
            static_hgraph = sample['static_hgraph']
            G = torch.zeros(size = (static_hgraph.shape[0], static_hgraph.shape[2], static_hgraph.shape[2])).to(device)
            for g in range(G.shape[0]):
                G[g] = torch.tensor(H2G(static_hgraph[g]), dtype = torch.float32)
            optimizer.zero_grad()
            output = model(sim_states, G).squeeze()
            patient_zero = patient_zero.squeeze()
            loss = criterion(output, patient_zero)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.4f}')

        if epoch % 5 == 0:
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, train_loader, device)
            logging.info(f'Train: MRR: {(mrr):.4f}, Hits@1: {(hits_at_1):.4f}, Hits@10: {(hits_at_10):.4f}, Hits@100: {(hits_at_100):.4f}')
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, valid_loader, device)
            logging.info(f'Valid: MRR: {(mrr):.4f}, Hits@1: {(hits_at_1):.4f}, Hits@10: {(hits_at_10):.4f}, Hits@100: {(hits_at_100):.4f}')
        

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
            sim_states = sim_states.reshape(-1, sim_states.size(2),  sim_states.size(3)).to(device)
            static_hgraph = sample['static_hgraph']
            G = torch.zeros(size = (static_hgraph.shape[0], static_hgraph.shape[2], static_hgraph.shape[2])).to(device)
            for g in range(G.shape[0]):
                G[g] = torch.tensor(H2G(static_hgraph[g]), dtype = torch.float32)
            output = model(sim_states, G).squeeze(dim=-1)

            # Ensure the output and target have matching dimensions
            batch_size, _ = output.size()

            for i in range(batch_size):
                target_indices = torch.nonzero(patient_zero[i]).contiguous().view(-1)
                _, indices = torch.sort(output[i], descending=True)
                ranks = torch.where(indices.unsqueeze(1) == target_indices)[0].float() + 1
                print(ranks)

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

def main():
    """
    Main file to run from the command line.
    """
    parser = argparse.ArgumentParser()

    #Log directory
    parser.add_argument("--log_path", type=str, default='test.log')

    #Training Args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--timestep_hidden", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--out_channels", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--mlp_layer", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    log_path = f'./{args.log_path}'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(args)
    logging.info('******************* Training Started *******************')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load('./data/sim#0/Sim_H2ABM.pt')
    print(data)

    horizon = data.hyperparameters['horizon']
    timestep_hidden = args.timestep_hidden

    x = data.sim_states
    x = x[:, timestep_hidden:, :, :]
    x = x.permute(0, 2, 1, 3) #Size([100, 2500, 3, 120])
    data.sim_states = x
    num_node = x.size(1)

    train_data, valid_data, test_data = split_dataset(data, seed = args.seed)

    train_data, valid_data, test_data = HypergraphDataset(train_data), HypergraphDataset(valid_data), HypergraphDataset(test_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    # model = SDSTGCN(N = 2500, 
    #                 in_channels = 360, 
    #                 out_channels = 256, 
    #                 num_blocks = 2, 
    #                 kernel_size = 3, 
    #                 spatial_out_channels = 256).to(device)

    model = THGNN(in_channels = 3,
                    out_channels = args.out_channels, 
                    hidden_size = args.hidden_size,
                    mlp_layer= args.mlp_layer,
                    num_node = num_node,
                    timesteps = horizon - timestep_hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train(model, train_loader, valid_loader, criterion, optimizer, args.epoch, logging, device)
    logging.info('******************* Training Finished *******************')

if __name__ == "__main__":
    main()