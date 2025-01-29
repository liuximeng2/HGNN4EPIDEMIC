import logging
import os
import argparse
from tqdm import tqdm
from model.dynamic_model import DTHGNN
from model.astgcn import ASTGCN
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from utils.hgraph_utils import *
from utils.utils import *

def train(model, train_loader, valid_loader, test_loader, optimizer, num_epochs, logging, device, args):

    best_valid_f1 = float('-inf')
    best_valid_acc = float('-inf')
    best_valid_auroc = float('-inf')
    best_valid_mse = float('inf')
    best_test_metrics = None

    for epoch in range(num_epochs + 1):
        model.train()
        flag = True
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            forecast_label = sample['forecast_label'].to(device).permute(0, 2, 1).float()

            # print(f"shape of sim_states {sim_states.shape}")
            # print(f"shape of forecast_label {forecast_label.shape}")
            
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']

            # print(f"type of the dynamic edge list {type(dynamic_edge_list)}")
            # print(f"length of the dynamic edge list {len(dynamic_edge_list)}")
            # print(f"length of first timestamp of dynamic edge list {len(dynamic_edge_list[0])}")

            optimizer.zero_grad()
            output = model(sim_states, dynamic_edge_list)

            # print(f"output shape: {output.shape}, forecast_label shape: {forecast_label}")

            num_positive = forecast_label.sum()
            num_negative = forecast_label.numel() - num_positive
            pos_weight = num_negative / num_positive

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            criterion.to(device)
            loss = criterion(output, forecast_label)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.3f}')
        print(f'Epoch {epoch}/{num_epochs}, Loss: {(loss.item()):.3f}')

        if epoch % 10 == 0:
            f1, acc, auroc, mse = evaluate_model(model, train_loader, device)
            logging.info(f'Train: F1: {(f1):.3f}, Acc: {(acc):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
            valid_f1, valid_acc, valid_auroc, valid_mse = evaluate_model(model, valid_loader, device)
            logging.info(f'Valid: F1: {(valid_f1):.3f}, Acc: {(valid_acc):.3f}, AUROC: {(valid_auroc):.3f}, MSE: {(valid_mse):.3f}')
            test_f1, test_acc, test_auroc, test_mse = evaluate_model(model, test_loader, device)
            logging.info(f'Test: F1: {(test_f1):.3f}, Acc: {(test_acc):.3f}, AUROC: {(test_auroc):.3f}, MSE: {(test_mse):.3f}')
            
            if valid_f1 > best_valid_f1:
                best_valid_f1 = test_f1
                best_valid_acc = test_acc
                best_valid_auroc = test_auroc
                best_valid_mse = test_mse
                best_test_metrics = {
                    'f1': best_valid_f1,
                    'acc': best_valid_acc,
                    'auroc': best_valid_auroc,
                    'mse': best_valid_mse
                }

    logging.info("Final Result:")
    logging.info(f'f1: {(best_test_metrics["f1"]):.3f}, acc: {(best_test_metrics["acc"]):.3f}, auroc: {(best_test_metrics["auroc"]):.3f}, mse: {(best_test_metrics["mse"]):.3f}')

def evaluate_model(model, data_loader, device):
    model.eval()
    total_samples = 0
    acc = 0
    auroc = 0
    mse = 0
    f1 = 0

    with torch.no_grad():
        for sample in data_loader:
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']
            sim_states = sample['sim_states'].to(device)
            forecast_label = sample['forecast_label'].to(device).permute(0, 2, 1).cpu().flatten().int()

            output = model(sim_states, dynamic_edge_list).cpu().flatten()  # Output shape: [batch_size, num_nodes, out_channel]

            output[output > 0.5] = 1
            output[output <= 0.5] = 0

            mse = (output.sum() - forecast_label.sum()) ** 2
            f1 += f1_score(output, forecast_label)
            acc += accuracy_score(output, forecast_label)
            # auroc += roc_auc_score(output, forecast_label)
            auroc = 0
            
            total_samples += 1

            
    return f1 / total_samples, acc / total_samples, auroc / total_samples, mse / total_samples


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
    parser.add_argument("--timestep_hidden", type=int, default=20)
    parser.add_argument("--known_interval", type=int, default=10)
    parser.add_argument("--pred_interval", type=int, default=5)

    args, _ = parser.parse_known_args()
    model_name = args.model

    model_args = model_dict[model_name]['default_args']
    for arg, default in model_args.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)
    
    parser.add_argument("--agg", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    log_path = f'./log/{args.dataset}/{args.model}/forecast'
    init_path(log_path)
    log_path += f'/tsh{args.timestep_hidden}-pre{args.pred_interval}-lr{args.lr}-b{args.batch_size}-drop{args.dropout}'
    log_path += f'-agg' if args.agg else ''
    log_path += f'-partial' if args.partial else ''
    log_path += '.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(now)
    logging.info(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    interested_interval = args.timestep_hidden + args.known_interval
    if args.dataset == 'UVA':
        if args.agg == True:
            logging.info('Aggregated Hypergraph')
            data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen_aggH.pt')
        if args.agg == False:
            logging.info('Sparse Hypergraph')
            data = torch.load('data/sim#0/DynamicSim_uva_ic1_pathogen.pt')
        data.forecast_label = data.sim_states[:, interested_interval:interested_interval + args.pred_interval, :, 1]
        data.dynamic_hypergraph = data.dynamic_hypergraph[0:interested_interval, :, :]
        data.dynamic_edge_list = process_hyperedges_incidence(data.dynamic_hypergraph, interested_interval)
        horizon = data.hyperparameters['horizon']
        assert data is not None, 'Data not found'
    
    if args.dataset == 'EpiSim':
        data = torch.load("data/epiSim/simulated_epi.pt")
        data.sim_states = data.sim_states.float()
        data.patient_zero = data.patient_zero.float()
        data.patient_zero = torch.unsqueeze(data.patient_zero, 1)
        data.dynamic_hypergraph = data.dynamic_hypergraph.float()
        horizon = data.sim_states.shape[1]
        pre_symptom = data.sim_states[:, interested_interval:interested_interval + args.pred_interval, :, 1]
        symptom = data.sim_states[:, interested_interval:interested_interval + args.pred_interval, :, 2]
        critical = data.sim_states[:, interested_interval:interested_interval + args.pred_interval, :, 3]
        #combine the three states into one using OR operation
        data.forecast_label = torch.logical_or(torch.logical_or(pre_symptom, symptom), critical)
        data.dynamic_hypergraph = data.dynamic_hypergraph[0:interested_interval, :, :]
        data.dynamic_edge_list = process_hyperedges_incidence(data.dynamic_hypergraph, interested_interval, multiple_instance=True)
        args.in_channels = 10
        print(len(data.dynamic_edge_list))
        
    print(data)

    assert horizon > args.timestep_hidden, 'Horizon should be greater than timestep_hidden'

    data.sim_states[:, 0:args.timestep_hidden, :, :] = 0 # mask the first timestep_hidden timesteps

    data.sim_states = data.sim_states[:, 0:interested_interval, :, :]
    args.len_input = interested_interval
    args.num_for_predict = args.pred_interval


    if args.model == 'ASTGCN' or args.model == 'MSTGCN':
        data.sim_states = data.sim_states.permute(0, 2 ,3 ,1)

    train_data, valid_data, test_data = split_dataset(data, seed=args.seed)

    if args.dataset == 'UVA':
        train_data, valid_data, test_data = DynamicHypergraphDataset(train_data), DynamicHypergraphDataset(valid_data), DynamicHypergraphDataset(test_data)
    if args.dataset == 'EpiSim':
        train_data, valid_data, test_data = DynamicHypergraphDataset(train_data, multiple_instance=True), DynamicHypergraphDataset(valid_data, multiple_instance=True), DynamicHypergraphDataset(test_data, multiple_instance=True)
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
