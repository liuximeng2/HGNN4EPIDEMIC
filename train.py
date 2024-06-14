import logging
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch_sparse import coalesce
from utils.hgraph_utils import *
from utils.utils import *

def train(model, train_loader, valid_loader, test_loader, optimizer, num_epochs, logging, device, args):

    best_valid_mrr = float('-inf')
    best_test_metrics = None
    flag = True

    for epoch in tqdm(range(num_epochs + 1)):
        model.train()
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            sim_states = sim_states.contiguous().view(-1, sim_states.size(2),  sim_states.size(3)).to(device)
            static_hgraph = sample['static_hgraph'][0]
            if args.struc == 'hgraph' and flag == True:
                #G = torch.tensor(H2G(static_hgraph), dtype = torch.float32).to(device)
                edge_index = H2EL(H2D(static_hgraph.T))
                total = edge_index.max() + 1
                edge_index, _ = coalesce(edge_index, None, total, total)
                edge_index = ExtractV2E(edge_index)
                edge_index[1] -= edge_index[1].min()
                edge_index = edge_index.to(device)
                flag = False
            if args.struc == 'graph' and flag == True:
                edge_index = Hyper2Graph(static_hgraph).to(device)
                flag = False
            optimizer.zero_grad()
            output = model(sim_states, edge_index).squeeze()
            patient_zero = patient_zero.squeeze()

            pos_loss = -torch.log(output[patient_zero > 0] + 1e-15).mean()
            neg_loss = -torch.log(1 - output[patient_zero == 0] + 1e-15).mean()
            loss = pos_loss * args.punishment + neg_loss

            #loss = criterion(output, patient_zero) * 10
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.3f}')

        if epoch % 5 == 0:
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, train_loader, device, args)
            logging.info(f'Train: MRR: {(mrr):.3f}, Hits@1: {(hits_at_1):.2f}, Hits@10: {(hits_at_10):.3f}, Hits@100: {(hits_at_100):.3f}')
            valid_mrr, valid_hits_at_1, valid_hits_at_10, valid_hits_at_100 = evaluate_model(model, valid_loader, device, args)
            logging.info(f'Valid: MRR: {(valid_mrr):.3f}, Hits@1: {(valid_hits_at_1):.3f}, Hits@10: {(valid_hits_at_10):.3f}, Hits@100: {(valid_hits_at_100):.3f}')
            test_mrr, test_hits_at_1, test_hits_at_10, test_hits_at_100 = evaluate_model(model, test_loader, device, args)
            logging.info(f'Test: MRR: {(test_mrr):.3f}, Hits@1: {(test_hits_at_1):.3f}, Hits@10: {(test_hits_at_10):.3f}, Hits@100: {(test_hits_at_100):.3f}')
            
            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                best_test_metrics = {
                    'MRR': test_mrr,
                    'Hits@1': test_hits_at_1,
                    'Hits@10': test_hits_at_10,
                    'Hits@100': test_hits_at_100
                }

    logging.info("Final Result:")
    logging.info(f'MRR: {(best_test_metrics["MRR"]):.3f}, Hits@1: {(best_test_metrics["Hits@1"]):.3f}, Hits@10: {(best_test_metrics["Hits@10"]):.3f}, Hits@100: {(best_test_metrics["Hits@100"]):.3f}')

def evaluate_model(model, data_loader, device, args):
    model.eval()
    mrr = 0
    hits_at_1 = 0
    hits_at_10 = 0
    hits_at_100 = 0
    total_samples = 0
    flag = True

    with torch.no_grad():
        for sample in data_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            sim_states = sim_states.reshape(-1, sim_states.size(2),  sim_states.size(3)).to(device)
            static_hgraph = sample['static_hgraph'][0]
            if args.struc == 'hgraph' and flag == True:
                #G = torch.tensor(H2G(static_hgraph), dtype = torch.float32).to(device)
                edge_index = H2EL(H2D(static_hgraph.T))
                total = edge_index.max() + 1
                edge_index, _ = coalesce(edge_index, None, total, total)
                edge_index = ExtractV2E(edge_index)
                edge_index[1] -= edge_index[1].min()
                edge_index = edge_index.to(device)
                flag = False
            if args.struc == 'graph' and flag == True:
                edge_index = Hyper2Graph(static_hgraph).to(device)
                flag = False
            output = model(sim_states, edge_index).squeeze()
            patient_zero = patient_zero.squeeze()

            # Ensure the output and target have matching dimensions
            batch_size = 1

            for i in range(batch_size):
                target_indices = torch.nonzero(patient_zero).contiguous().view(-1)
                _, indices = torch.sort(output, descending=True)
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

def main():
    """
    Main file to run from the command line.
    """
    from config import model_dict

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='random')
    parser.add_argument("--model", type=str, default='TGCN', choices=model_dict.keys())

    args, _ = parser.parse_known_args()
    model_name = args.model

    model_args = model_dict[model_name]['default_args']
    for arg, default in model_args.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)

    args = parser.parse_args()
    model = model_dict[model_name]['class']

    set_seed(args.seed)
    log_path = f'./log/{args.dataset}/{args.model}'
    init_path(log_path)
    log_path += f'/tsh{args.timestep_hidden}_oc{args.out_channels}_lr{args.lr}-b{args.batch_size}.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(args)
    logging.info('******************* Training Started *******************')

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'random':
        data = torch.load('./data/sim#0/n2500_random_ic1.pt')
    if args.dataset == 'h2abm':
        data = torch.load('./data/sim#0/Sim_H2ABM_ic1.pt')
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
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    model = model(in_channels = 3,
                    num_node = num_node,
                    timesteps = horizon - timestep_hidden,
                    args = args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, valid_loader, test_loader, optimizer, args.epoch, logging, device, args)
    logging.info('******************* Training Finished *******************')

if __name__ == "__main__":
    main()