import logging
import os
import argparse
from tqdm import tqdm
from model.dynamic_model import DTHGNN
from model.astgcn import ASTGCN

import torch
import torch.nn as nn
from utils.hgraph_utils import *
from utils.utils import *

def train(model, train_loader, valid_loader, test_loader, optimizer, num_epochs, logging, device, args):

    best_valid_mrr = float('-inf')
    best_test_metrics = None

    for epoch in range(num_epochs + 1):
        model.train()
        flag = True
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']

            optimizer.zero_grad()
            output = model(sim_states, dynamic_edge_list)

            target = torch.zeros_like(output)
            for i in range(args.batch_size):
                target[i, patient_zero[i]] = 1

            num_positive = target.sum()
            num_negative = target.numel() - num_positive
            pos_weight = num_negative / num_positive

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            criterion.to(device)
            # print(f"output shape: {output.shape}, target shape: {target.shape}")
            loss = criterion(output, target)

            loss.backward()
            # if flag == True:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print(f"{name} grad mean: {param.grad.mean()}, grad std: {param.grad.std()}")
            #     flag = False
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.3f}')
        print(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.3f}')

        if epoch % 10 == 0:
            mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, train_loader, device)
            logging.info(f'Train: MRR: {(mrr):.3f}, Hits@1: {(hits_at_1):.3f}, Hits@10: {(hits_at_10):.3f}, Hits@100: {(hits_at_100):.3f}')
            valid_mrr, valid_hits_at_1, valid_hits_at_10, valid_hits_at_100 = evaluate_model(model, valid_loader, device)
            logging.info(f'Valid: MRR: {(valid_mrr):.3f}, Hits@1: {(valid_hits_at_1):.3f}, Hits@10: {(valid_hits_at_10):.3f}, Hits@100: {(valid_hits_at_100):.3f}')
            test_mrr, test_hits_at_1, test_hits_at_10, test_hits_at_100 = evaluate_model(model, test_loader, device)
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

def evaluate_model(model, data_loader, device):
    model.eval()
    total_mrr = 0.0
    total_hits_at_1 = 0.0
    total_hits_at_10 = 0.0
    total_hits_at_100 = 0.0
    total_samples = 0

    with torch.no_grad():
        for sample in data_loader:
            # Remove [0] to include all samples in the batch
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']
            sim_states = sample['sim_states'].to(device)
            patient_zero = sample['patient_zero'].to(device)  # Shape: [batch_size]

            output = model(sim_states, dynamic_edge_list)  # Output shape: [batch_size, num_nodes]
            batch_size, num_nodes, out_channle = output.shape

            # Calculate rankings for each sample in the batch
            for i in range(batch_size):
                output_scores = output[i].squeeze(dim=1)  # Shape: [num_nodes]
                true_index = patient_zero[i]  # Scalar
                print(f"output_scores shape: {output_scores.shape}, true_index shape: {true_index.shape}")
                os.exit(0)

                # Sort the scores in descending order
                _, indices = torch.sort(output_scores, descending=True)
                # Find the rank of the true index (add 1 for 1-based ranking)
                rank = (indices == true_index).nonzero(as_tuple=False).item() + 1

                # Update metrics
                total_mrr += 1.0 / rank
                total_hits_at_1 += 1.0 if rank <= 1 else 0.0
                total_hits_at_10 += 1.0 if rank <= 10 else 0.0
                total_hits_at_100 += 1.0 if rank <= 100 else 0.0
                total_samples += 1

    # Compute average metrics
    mrr = total_mrr / total_samples
    hits_at_1 = total_hits_at_1 / total_samples
    hits_at_10 = total_hits_at_10 / total_samples
    hits_at_100 = total_hits_at_100 / total_samples

    return mrr, hits_at_1, hits_at_10, hits_at_100


def main():
    """
    Main file to run from the command line.
    """

    from config import model_dict
    import datetime
    now = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dataset", type=str, default='UVA')
    parser.add_argument("--model", type=str, default='DTHGNN', choices=model_dict.keys())
    parser.add_argument("--struct", type=str, default='RNN-GNN')
    parser.add_argument("--timestep_hidden", type=int, default=20)
    parser.add_argument("--known_interval", type=int, default=10)
    parser.add_argument("--pred_interval", type=int, default=10)

    args, _ = parser.parse_known_args()
    model_name = args.model

    model_args = model_dict[model_name]['default_args']
    for arg, default in model_args.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)

    parser.add_argument("--agg", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    log_path = f'./log/UVA/{args.model}'
    init_path(log_path)
    log_path += f'/tsh{args.timestep_hidden}-lr{args.lr}-b{args.batch_size}-drop{args.dropout}'
    log_path += f'-agg' if args.agg else ''
    log_path += f'-partial' if args.partial else ''
    log_path += '.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(now)
    logging.info(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.agg == True:
        logging.info('Aggregated Hypergraph')
        data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen_aggH.pt')
    if args.agg == False:
        logging.info('Sparse Hypergraph')
        data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen.pt')

    horizon = data.hyperparameters['horizon']
    assert data is not None, 'Data not found'
    assert horizon > args.timestep_hidden, 'Horizon should be greater than timestep_hidden'

    data.sim_states[:, 0:args.timestep_hidden, :, :] = 0 # mask the first timestep_hidden timesteps
    args.len_input = horizon

    interested_interval = args.timestep_hidden + args.known_interval
    data.forecast_label = data.sim_states[:, interested_interval:interested_interval + args.pred_interval, :, 1]
    data.sim_states = data.sim_states[:, 0:interested_interval, :, :]
    data.dynamic_hypergraph = data.dynamic_hypergraph[0:interested_interval, :, :]
    data.dynamic_edge_list = process_hyperedges_incidence(data.dynamic_hypergraph, interested_interval)
    args.len_input = interested_interval

    if args.model == 'ASTGCN' or args.model == 'MSTGCN':
        data.sim_states = data.sim_states.permute(0, 2 ,3 ,1)

    train_data, valid_data, test_data = split_dataset(data, seed=args.seed)

    train_data, valid_data, test_data = DynamicHypergraphDataset(train_data), DynamicHypergraphDataset(valid_data), DynamicHypergraphDataset(test_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = model_dict[model_name]['class']
    model = model(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logging.info('******************* Training Started *******************')
    print('******************* Training Started *******************')
    train(model, train_loader, valid_loader, test_loader, optimizer, args.epochs, logging, device, args)
    logging.info('******************* Training Finished *******************')

if __name__ == "__main__":
    main()