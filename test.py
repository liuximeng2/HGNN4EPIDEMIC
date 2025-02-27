import logging
import os
import argparse
import itertools
import datetime
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from utils.hgraph_utils import *
from utils.utils import *
from config import model_dict

def train(model, train_loader, valid_loader, test_loader, optimizer, num_epochs, device, args):
    """Train the model for a specified number of epochs."""
    best_valid_metric = float('-inf')
    best_test_metrics = None

    for epoch in range(num_epochs + 1):
        model.train()
        for sample in train_loader:
            sim_states, patient_zero = sample['sim_states'].to(device), sample['patient_zero'].to(device)
            forecast_label = sample['forecast_label'].to(device).permute(0, 2, 1).float()
            dynamic_edge_list = [edge[0].to(device) for edge in sample['dynamic_edge_list']]
            
            optimizer.zero_grad()

            if args.model == 'DTHGNN':
                output, recon_graph, pos_recon_logit, neg_recon_logit = model(sim_states, dynamic_edge_list)
                pos_loss = -torch.log(pos_recon_logit + 1e-15).mean()
                neg_loss = -torch.log(1 - neg_recon_logit + 1e-15).mean()
                recon_loss = pos_loss + neg_loss
            else:
                output = model(sim_states, dynamic_edge_list).to(device)
            
            if args.forecast:
                output = nn.Sigmoid()(output)
                postive_scores = output[0, torch.where(forecast_label[0, :, :] == 1)[0], torch.where(forecast_label[0, :, :] == 1)[1]]
                negative_scores = output[0, torch.where(forecast_label[0, :, :] == 0)[0], torch.where(forecast_label[0, :, :] == 0)[1]]
                # scale down negative scores to have the same number of samples as positive scores randomly
                negative_scores = negative_scores[torch.randperm(len(negative_scores))[:len(postive_scores)]]
                pos_forecast_loss = -torch.log(postive_scores + 1e-15).mean()
                neg_forecast_loss = -torch.log(1 - negative_scores + 1e-15).mean()
                forecast_loss = pos_forecast_loss + neg_forecast_loss
                mae_loss = torch.abs(output.sum() - forecast_label.sum()) / 4000

            else:
                target = torch.zeros_like(output)
                for i in range(args.batch_size):
                    target[i, patient_zero[i]] = 1
            
            if args.forecast:
                loss = args.alpha * forecast_loss + (1 - args.alpha) * recon_loss if args.location_aware else forecast_loss + mae_loss
            else:
                pos_weight = (target.numel() - target.sum()) / target.sum()
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
                indiv_loss = criterion(output, target)
                loss = args.alpha * indiv_loss + (1 - args.alpha) * recon_loss if args.location_aware else indiv_loss
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if args.forecast:
            logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}, Forecast Loss: {forecast_loss.item():.3f}, MAE Loss: {mae_loss.item():.3f}')
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}, Forecast Loss: {forecast_loss.item():.3f}, MAE Loss: {mae_loss.item():.3f}')
        else:
            logging.info(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}, Recon Loss: {recon_loss.item():.3f}, Indiv Loss: {indiv_loss.item():.3f}')
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.3f}, Recon Loss: {recon_loss.item():.3f}, Indiv Loss: {indiv_loss.item():.3f}')

        if epoch % 5 == 0:
            if not args.forecast:
                mrr, hits_at_1, hits_at_10, hits_at_100 = evaluate_model(model, train_loader, device, args)
                logging.info(f'Train: MRR: {(mrr):.3f}, Hits@1: {(hits_at_1):.3f}, Hits@10: {(hits_at_10):.3f}, Hits@100: {(hits_at_100):.3f}')
                print(f'Train: MRR: {(mrr):.3f}, Hits@1: {(hits_at_1):.3f}, Hits@10: {(hits_at_10):.3f}, Hits@100: {(hits_at_100):.3f}')
                valid_mrr, valid_hits_at_1, valid_hits_at_10, valid_hits_at_100 = evaluate_model(model, valid_loader, device, args)
                logging.info(f'Valid: MRR: {(valid_mrr):.3f}, Hits@1: {(valid_hits_at_1):.3f}, Hits@10: {(valid_hits_at_10):.3f}, Hits@100: {(valid_hits_at_100):.3f}')
                print(f'Valid: MRR: {(valid_mrr):.3f}, Hits@1: {(valid_hits_at_1):.3f}, Hits@10: {(valid_hits_at_10):.3f}, Hits@100: {(valid_hits_at_100):.3f}')
                test_mrr, test_hits_at_1, test_hits_at_10, test_hits_at_100 = evaluate_model(model, test_loader, device, args)
                logging.info(f'Test: MRR: {(test_mrr):.3f}, Hits@1: {(test_hits_at_1):.3f}, Hits@10: {(test_hits_at_10):.3f}, Hits@100: {(test_hits_at_100):.3f}')
                print(f'Test: MRR: {(test_mrr):.3f}, Hits@1: {(test_hits_at_1):.3f}, Hits@10: {(test_hits_at_10):.3f}, Hits@100: {(test_hits_at_100):.3f}')
                
                if valid_mrr > best_valid_metric:
                    best_valid_metric = valid_mrr
                    best_mrr = test_mrr
                    best_test_metrics = {
                        'MRR': test_mrr,
                        'Hits@1': test_hits_at_1,
                        'Hits@10': test_hits_at_10,
                        'Hits@100': test_hits_at_100
                    }

            
            if args.forecast:
                f1, auroc, mse = evaluate_forecast_model(model, train_loader, device, args)
                logging.info(f'Train: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
                print(f'Train: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
                valid_f1, auroc, mse = evaluate_forecast_model(model, valid_loader, device, args)
                logging.info(f'Valid: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
                print(f'Valid: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
                f1, auroc, mse = evaluate_forecast_model(model, test_loader, device, args)
                logging.info(f'Test: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')
                print(f'Test: F1: {(f1):.3f}, AUROC: {(auroc):.3f}, MSE: {(mse):.3f}')

                
                if valid_f1 > best_valid_metric:
                    best_valid_metric = valid_f1
                    best_f1 = f1
                    best_test_metrics = {
                        'F1': f1,
                        'AUROC': auroc,
                        'MSE': mse
                    }

    if args.forecast:
        return best_f1, best_test_metrics
    else:
        return best_mrr, best_test_metrics
    


def evaluate_forecast_model(model, data_loader, device, args):
    model.eval()
    total_samples = 0
    auroc = 0
    mse = 0
    f1 = 0
    daily_stats = {i: {'f1': 0, 'auroc': 0, 'mae': 0} for i in range(args.num_for_predict)}

    with torch.no_grad():
        for sample in data_loader:
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']
            sim_states = sample['sim_states'].to(device)
            forecast_label = sample['forecast_label'].to(device).permute(0, 2, 1).cpu().flatten().int()

            if args.model == 'DTHGNN':
                output, _, _,_ = model(sim_states, dynamic_edge_list)
            else:
                output = model(sim_states, dynamic_edge_list).to(device)
            output = nn.Sigmoid()(output).cpu().flatten()

            output[output > 0.8] = 1
            output[output <= 0.8] = 0
            
            # Ensure compatibility with binary classification
            output[0] = 0 if output[1] > 0.5 else 1
            forecast_label[0] = 0 if forecast_label[1] > 0.5 else 1

            mse += abs(output.sum() - forecast_label.sum())
            f1 += f1_score(output, forecast_label)
            auroc += roc_auc_score(output, forecast_label)

            per_time_step = output.reshape(-1, args.num_of_vertices)
            per_time_step = per_time_step.sum(dim = 1)
            logging.info(f"Prediction of sample {total_samples}: {per_time_step}")
            
            total_samples += 1

    #         for i in range(args.num_for_predict):
    #             day_i_ouput = output[i * args.num_of_vertices: (i + 1) * args.num_of_vertices]
    #             day_i_label = forecast_label[i * args.num_of_vertices: (i + 1) * args.num_of_vertices]
    #             daily_stats[i]['f1'] += f1_score(day_i_label, day_i_ouput)
    #             daily_stats[i]['mae'] += abs(day_i_ouput.sum() - day_i_label.sum())
    #             daily_stats[i]['auroc'] += roc_auc_score(day_i_label, day_i_ouput)

    # daily_stats = {i: {k: v / total_samples for k, v in stats.items()} for i, stats in daily_stats.items()}
    # print(daily_stats)

    return f1 / total_samples,  auroc / total_samples, mse / total_samples

def evaluate_model(model, data_loader, device, args):
    model.eval()
    total_mrr = 0.0
    total_hits_at_1 = 0.0
    total_hits_at_10 = 0.0
    total_hits_at_100 = 0.0
    total_samples = 0

    with torch.no_grad():
        for sample in data_loader:
            for i in range(len(sample['dynamic_edge_list'])):
                sample['dynamic_edge_list'][i] = sample['dynamic_edge_list'][i][0].to(device)
            dynamic_edge_list = sample['dynamic_edge_list']
            sim_states = sample['sim_states'].to(device)
            patient_zero = sample['patient_zero'].to(device)
            dynamic_graph = sample['dynamic_hgraph'].to(device)


            if args.model == 'DTHGNN':
                output, recon_graph, _, _ = model(sim_states, dynamic_edge_list)
                ###########
                recon_graph_binary = (recon_graph <= 0.5).float()  # Ensure the flip works correctly
                recon_graph_binary = recon_graph_binary.squeeze().long()
                dgraph = dynamic_graph[0][-1].long().cpu()
                
                interaction_intensity = dgraph.sum(dim=1).float().cpu()
                print(torch.quantile(interaction_intensity, torch.tensor([0.0, 0.25, 0.5, 0.75, 1])))

                quantiles = torch.quantile(interaction_intensity, torch.tensor([0.0, 0.25, 0.5, 0.75, 1]))
                # print("Quantiles:", quantiles)
                # compute location f1 score
                location_accuracy = (recon_graph_binary == dgraph).float().mean(dim=1).cpu()

                # select indices of the quantiles calculation and compute accuracy for each bin
                for i in range(len(quantiles) - 1):
                    low = quantiles[i]
                    high = quantiles[i + 1]
                    # include the left edge and right edge for the last bin
                    if i < len(quantiles) - 2:
                        idx = (interaction_intensity >= low) & (interaction_intensity < high)
                    else:
                        idx = (interaction_intensity >= low) & (interaction_intensity <= high)
                    
                    if idx.sum() > 0:
                        bin_accuracy = location_accuracy[idx].mean().item()
                        bin_f1 = f1_score(recon_graph_binary[idx].cpu().flatten(), dgraph[idx].cpu().flatten())
                        print(f"f1 for quantile bin {i} ({low:.3f}-{high:.3f}): {bin_f1:.3f}")
                        print(f"accuracy for quantile bin {i} ({low:.3f}-{high:.3f}): {bin_accuracy:.3f}")
                    else:
                        print(f"No samples found in quantile bin {i} ({low:.3f}-{high:.3f})")
                ###########
            else:
                output = model(sim_states, dynamic_edge_list).to(device)
            batch_size, num_nodes, out_channle = output.shape

            # Calculate rankings for each sample in the batch
            for i in range(batch_size):
                output_scores = output[i].squeeze(dim=1)  # Shape: [num_nodes]
                true_index = patient_zero[i]  # Scalar

                # Sort the scores in descending order
                _, indices = torch.sort(output_scores, descending=True)
                # Find the rank of the true index (add 1 for 1-based ranking)
                rank = (indices == true_index).nonzero(as_tuple=False).item() + 1

                total_mrr += 1.0 / rank
                total_hits_at_1 += 1.0 if rank <= 1 else 0.0
                total_hits_at_10 += 1.0 if rank <= 3 else 0.0
                total_hits_at_100 += 1.0 if rank <= 10 else 0.0
                total_samples += 1

    # Compute average metrics
    mrr = total_mrr / total_samples
    hits_at_1 = total_hits_at_1 / total_samples
    hits_at_10 = total_hits_at_10 / total_samples
    hits_at_100 = total_hits_at_100 / total_samples

    return mrr, hits_at_1, hits_at_10, hits_at_100

def hyperparameter_search(args, model_name, parser, train_loader, valid_loader, test_loader, device):
    """Perform hyperparameter tuning using grid search."""
    param_grid = {
        'hidden_channels': [256],
        'lr': [1e-3, 1e-4, 1e-5, 1e-6],
        'alpha': [0.5, 0.7, 0.9],
        'weight_decay': [1e-2, 1e-3, 1e-4],
        'kernal_size': [2, 4, 8]
    }
    param_combinations = list(itertools.product(*param_grid.values()))
    best_config, best_performance = None, float('-inf')

    for params in param_combinations:
        print("Currently testing:", params)
        hidden_channels, lr, alpha = params
        args.hidden_channels = hidden_channels
        args.lr = lr
        args.alpha = alpha
        model = model_dict[args.model]['class'](args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        epoch = 20
        
        best_mrr, _ = train(model, train_loader, valid_loader, test_loader, optimizer, epoch, device, args)
        
        if best_mrr > best_performance:
            best_performance = best_mrr
            best_config = params
    
    logging.info(f'Best Hyperparameters: {best_config}, Best Performance: {best_performance}')
    return best_config

def main():
    """Main function for training and hyperparameter tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--dataset", type=str, default='UVA')
    parser.add_argument("--model", type=str, default='DTHGNN', choices=model_dict.keys())
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--timestep_hidden", type=int, default=20)
    parser.add_argument("--known_interval", type=int, default=10)
    parser.add_argument("--location_aware", action="store_true")
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--num_for_predict", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)

    # Add tuned Hyperparameters
    parser.add_argument("--hidden_channels", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--kernal_size", type=int)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    model_args = model_dict[args.model]['default_args']

    for arg, default in model_args.items():
        parser.add_argument(f"--{arg}", type=type(default), default=default)
    args = parser.parse_args()

    seed = 0
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    interested_interval = args.timestep_hidden + args.known_interval
    data, args = load_data(args, interested_interval)

    if args.model == 'ASTGCN' or args.model == 'MSTGCN' or args.model == 'MSTHGNN':
        data.sim_states = data.sim_states.permute(0, 2 ,3 ,1)

    train_loader, valid_loader, test_loader = create_dataloader(args, data, seed)
    
    log_path = f'./log/{args.dataset}/{args.model}/'
    log_path += 'detect' if not args.forecast else 'forecast'
    init_path(log_path)
    if args.forecast:
        log_path += f'/ps{args.num_for_predict}-hidden{args.hidden_channels}-lr{args.lr}-wd{args.weight_decay}-kernal{args.kernal_size}-alpha{args.alpha}'
    else:
        log_path += f'/tsh{args.timestep_hidden}-hidden{args.hidden_channels}-lr{args.lr}-wd{args.weight_decay}-kernal{args.kernal_size}-alpha{args.alpha}'
    log_path += 'loc.log' if args.location_aware else '.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(args)
    print(args)
    
    # best_params = hyperparameter_search(args, args.model, parser, train_loader, valid_loader, test_loader, device)
    # logging.info("Search complete. Best hyperparameters:", best_params)
    
    results = []
    for run in range(args.runs):
        set_seed(run)
        # args.hidden_channels, args.lr, args.alpha = best_params
        model = model_dict[args.model]['class'](args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # train_loader, valid_loader, test_loader = create_dataloader(args, data, seed=run)
        
        _, final_performance = train(model, train_loader, valid_loader, test_loader, optimizer, args.epochs, device, args)
        results.append(final_performance)
    

    avg_performance = {metric: sum(d[metric] for d in results) / len(results) for metric in results[0]}
    std_dev = {metric: torch.std(torch.tensor([d[metric] for d in results])).item() for metric in results[0]}
    
    logging.info(f'Final Average Performance: {avg_performance}, Standard Deviation: {std_dev}')
    if args.forecast:
        logging.info(f'F1: {avg_performance["F1"]:.3f} +- {std_dev["F1"]:.3f}')
        logging.info(f'AUROC: {avg_performance["AUROC"]:.3f} +- {std_dev["AUROC"]:.3f}')
        logging.info(f'MSE: {avg_performance["MSE"]:.3f} +- {std_dev["MSE"]:.3f}')
    else:
        logging.info(f'MRR: {avg_performance["MRR"]:.3f} +- {std_dev["MRR"]:.3f}')
        logging.info(f'Hit@1: {avg_performance["Hits@1"]:.3f} +- {std_dev["Hits@1"]:.3f}')
        logging.info(f'Hit@3: {avg_performance["Hits@10"]:.3f} +- {std_dev["Hits@10"]:.3f}')
        logging.info(f'Hit@10: {avg_performance["Hits@100"]:.3f} +- {std_dev["Hits@100"]:.3f}')

if __name__ == "__main__":
    main()
