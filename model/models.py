import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model.layers import *
    
class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_out_channels):
        super(STConvBlock, self).__init__()
        self.temporal1 = GRU(in_channels, out_channels)
        self.spatial = GCN(out_channels, spatial_out_channels, dropout=0.0)
        self.temporal2 = GRU(spatial_out_channels, out_channels)
    
    def forward(self, x, G):

        x = self.temporal1(x)
        x = self.spatial(x, G)
        x = self.temporal2(x)

        return x

class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer 
    is implemented in pyg.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(HCHA, self).__init__()

        self.dropout = dropout
        self.symdegnorm = True
        self.invest = 1

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(HypergraphConv(input_dim, output_dim, self.symdegnorm))

        elif num_layers > 1:
            self.convs.append(HypergraphConv(input_dim, hidden_dim, self.symdegnorm))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    HypergraphConv(hidden_dim, hidden_dim, self.symdegnorm))
            self.convs.append(HypergraphConv(hidden_dim, output_dim, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):

        if self.invest == 1:
            print('layers in hgcn: ', len(self.convs))
            self.invest = 0

        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 1, dropout = 0.0):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))

        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout
       
        self.invest = 1

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
     
    def forward(self, x, adj_t):

        if self.invest == 1:
            print('layers in gcn: ', len(self.convs))
            self.invest = 0
            
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        return x

class STGCN(nn.Module):
    def __init__(self, N, in_channels, out_channels, num_blocks, spatial_out_channels):
        super(STGCN, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(STConvBlock(in_channels, out_channels, spatial_out_channels))
        
        for _ in range(num_blocks - 1):
            self.blocks.append(STConvBlock(out_channels, out_channels, spatial_out_channels))
        
        self.final_temporal = nn.GRU(out_channels, out_channels, num_layers=1, batch_first = True)
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, G):
        for block in self.blocks:
            x = block(x, G)
        
        x, _ = self.final_temporal(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class THGNN(nn.Module):
    def __init__(self, in_channels, num_node, timesteps, args):
        super(THGNN, self).__init__()
        self.num_node = num_node
        self.timesteps = timesteps
        self.out_channels = args.out_channels
        self.temporal = GRU(in_channels, args.out_channels, args.temporal_layer, dropout=args.dropout)
        self.spatial = HCHA(timesteps * args.out_channels, args.hidden_size, args.hidden_size, args.spatial_layer, dropout=args.dropout)
        self.fc = MLP(args.hidden_size, args.hidden_size, 1, num_layers=args.mlp_layer, dropout=args.dropout)
    
    def forward(self, x, edge_index):
        x = self.temporal(x)
        x = x.reshape(self.num_node, self.timesteps * self.out_channels)
        x = F.relu(self.spatial(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x)
    
class TGCN(nn.Module):
    def __init__(self, in_channels, num_node, timesteps, args):
        super(TGCN, self).__init__()
        self.num_node = num_node
        self.timesteps = timesteps
        self.out_channels = args.out_channels
        self.temporal = GRU(in_channels, args.out_channels, args.temporal_layer, dropout=args.dropout)
        self.spatial = GCN(timesteps * args.out_channels, args.hidden_size, args.hidden_size, dropout=args.dropout)
        self.fc = MLP(args.hidden_size, args.hidden_size, 1, num_layers=args.mlp_layer, dropout=args.dropout)
        
    def forward(self, x, edge_index):
        x = self.temporal(x)
        x = x.reshape(-1, self.num_node, self.timesteps * self.out_channels)
        x = F.relu(self.spatial(x, edge_index))
        x = self.fc(x)
        return torch.sigmoid(x)
    
class TMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, mlp_layer, temporal_layer, num_node, timesteps):
        super(TMLP, self).__init__()
        self.num_node = num_node
        self.timesteps = timesteps
        self.out_channels = out_channels
        self.temporal1 = GRU(in_channels, out_channels, temporal_layer)
        self.spatial = nn.Linear(timesteps * out_channels, hidden_size)
        self.fc = MLP(hidden_size, hidden_size, 1, num_layers=mlp_layer)
        
    def forward(self, x, G):
        x = self.temporal1(x)
        x = x.reshape(-1, self.num_node, self.timesteps * self.out_channels)
        x = F.relu(self.spatial(x))
        x = self.fc(x)
        return torch.sigmoid(x)