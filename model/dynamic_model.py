import os
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv
from model.layers import HypergraphConv, MLP
from utils.hgraph_utils import *
from utils.utils import *

from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms import LaplacianLambdaMax
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian

class DTHGNN(nn.Module):
    def __init__(self, args):
        super(DTHGNN, self).__init__()

        self.gru1 = nn.GRU(args.in_channels, args.out_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.gru2 = nn.GRU(args.hidden_channels, args.hidden_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.gnn = HypergraphConv(args.out_channels, args.hidden_channels, symdegnorm=True, use_attention=False, dropout=args.dropout)
        self.layer_norm1 = nn.LayerNorm(args.out_channels)

        self.layer_norm2 = nn.LayerNorm(args.hidden_channels)
        self.mlp = MLP(args.hidden_channels, args.hidden_channels, 1, args.mlp_layer, args.dropout)
        self.sigmoid = nn.Sigmoid()
        self._final_conv = nn.Conv2d(
            int(args.len_input / args.time_strides),
            args.num_for_predict,
            kernel_size=(1, args.nb_time_filter),
        )

    def forward(self, node_features, dynamic_edge_list):
        """
        Args:
            hyperedges_list (list of torch.Tensor): List of incidence matrices for each timestep.
                Each incidence matrix has shape [num_hyperedge, num_node].
            node_features (torch.Tensor): Node feature tensor of shape [batch_size, timestep, num_node, in_channels].
        Returns:
            torch.Tensor: Encoded node features of shape [batch_size, num_node, 1].
        """
        batch_size, timestep, num_node, in_channels = node_features.shape

        # Process node features with the first GRU

        # Reshape for GRU: [batch_size * num_node, timestep, in_channels]
        node_features = node_features.permute(0, 2, 1, 3).reshape(batch_size * num_node, timestep, in_channels)
        node_features, _ = self.gru1(node_features)
        node_features = node_features.reshape(batch_size * num_node, timestep, 2, -1)
        node_features = self.layer_norm1(node_features)
        node_features = node_features[:, :, 1, :] # insterested in the backward direction
        # Reshape back: [batch_size, num_node, timestep, out_channels]
        node_features = node_features.reshape(batch_size, num_node, timestep, -1)

        embeddings_over_time = []
        for t in range(timestep):

            edge_index = dynamic_edge_list[t]
            x = node_features[:, :, t, :].reshape(batch_size * num_node, -1) # Get node features at time t: [batch_size, num_node, in_channels]

            # edge_index = self.process_hyperedges_incidence(H)
            x = self.gnn(x, edge_index)
            # Reshape x to [batch_size, num_node, hidden_channels]
            x = x.reshape(batch_size, num_node, -1)
            embeddings_over_time.append(x)

        # Stack embeddings over time: [batch_size, timestep, num_node, hidden_channels]
        embeddings_over_time = torch.stack(embeddings_over_time, dim=1)

        # Pass through the second GRU
        #embeddings_over_time = embeddings_over_time.reshape(batch_size * num_node, timestep, -1)
        output = self._final_conv(embeddings_over_time)
        output = output[:, :, :, -1] # (b,c_out*T,N) for example (32, 12, 307)
        output = output.permute(0, 2, 1) # (b,T,N)-> (b,N,T)
        
        return self.sigmoid(output) #(b,N,T) for exmaple (32, 307,12)
    


class DTGNN(nn.Module):
    def __init__(self, args):
        super(DTGNN, self).__init__()

        self.gru1 = nn.GRU(args.in_channels, args.out_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.gru2 = nn.GRU(args.hidden_channels, args.hidden_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        # self.gnn = GCNConv(args.out_channels, args.hidden_channels)
        self.gnn = ChebConv(args.out_channels, out_channels=args.hidden_channels, K=3)
        self.layer_norm1 = nn.LayerNorm(args.out_channels)

        self.layer_norm2 = nn.LayerNorm(args.hidden_channels)
        self.mlp = MLP(args.hidden_channels, args.hidden_channels, 1, args.mlp_layer, args.dropout)

        self._final_conv = nn.Conv2d(
            int(args.len_input / args.time_strides),
            args.num_for_predict,
            kernel_size=(1, args.nb_time_filter),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, dynamic_edge_list):
        """
        Args:
            hyperedges_list (list of torch.Tensor): List of incidence matrices for each timestep.
                Each incidence matrix has shape [num_hyperedge, num_node].
            node_features (torch.Tensor): Node feature tensor of shape [batch_size, timestep, num_node, in_channels].
        Returns:
            torch.Tensor: Encoded node features of shape [batch_size, num_node, 1].
        """
        batch_size, timestep, num_node, in_channels = node_features.shape

        # Reshape for GRU: [batch_size * num_node, timestep, in_channels]
        node_features = node_features.permute(0, 2, 1, 3).reshape(batch_size * num_node, timestep, in_channels)
        node_features, _ = self.gru1(node_features)
        node_features = node_features.reshape(batch_size * num_node, timestep, 2, -1)
        node_features = self.layer_norm1(node_features)
        node_features = node_features[:, :, 1, :] # insterested in the backward direction
        # Reshape back: [batch_size, num_node, timestep, out_channels]
        node_features = node_features.reshape(batch_size, num_node, timestep, -1)

        embeddings_over_time = []
        for t in range(timestep):

            # Get node features at time t: [batch_size, num_node, in_channels]
            x = node_features[:, :, t, :].reshape(batch_size * num_node, -1)
            
            edge_index = dynamic_edge_list[t].to(node_features.device)
            lambda_max = LaplacianLambdaMax()(
                    Data(
                        edge_index=edge_index,
                        edge_attr=None,
                        num_nodes=num_node,
                    )
                ).lambda_max
            x = self.gnn(x, edge_index, lambda_max=lambda_max)
            # Reshape x to [batch_size, num_node, hidden_channels]
            x = x.reshape(batch_size, num_node, -1)
            embeddings_over_time.append(x)

        # Stack embeddings over time: [batch_size, timestep, num_node, hidden_channels]
        embeddings_over_time = torch.stack(embeddings_over_time, dim=1)

        output = self._final_conv(embeddings_over_time)
        output = output[:, :, :, -1] # (b,c_out*T,N) for example (32, 12, 307)
        output = output.permute(0, 2, 1) # (b,T,N)-> (b,N,T)

        return self.sigmoid(output) #(b,N,T) for exmaple (32, 307,12)

class FDTHGNN(nn.Module):
    def __init__(self, args):
        super(FDTHGNN, self).__init__()
        self.gru1 = nn.GRU(args.in_channels, args.out_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.gru2 = nn.GRU(args.hidden_channels, args.hidden_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        self.gnn = HypergraphConv(args.out_channels, args.hidden_channels, symdegnorm=True, use_attention=False, dropout=args.dropout)
        self.layer_norm1 = nn.LayerNorm(args.out_channels)

        self.layer_norm2 = nn.LayerNorm(args.hidden_channels)
        self.mlp = MLP(args.hidden_channels, args.hidden_channels, 1, args.mlp_layer, args.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._final_conv = nn.Conv2d(
            int(args.len_input / args.time_strides),
            args.pred_interval,
            kernel_size=(1, args.nb_time_filter),
        )

    def forward(self, node_features, dynamic_edge_list):
        """
        Args:
            hyperedges_list (list of torch.Tensor): List of incidence matrices for each timestep.
                Each incidence matrix has shape [num_hyperedge, num_node].
            node_features (torch.Tensor): Node feature tensor of shape [batch_size, timestep, num_node, in_channels].
        Returns:
            torch.Tensor: Encoded node features of shape [batch_size, num_node, 1].
        """
        batch_size, timestep, num_node, in_channels = node_features.shape

        # Process node features with the first GRU
        # Reshape for GRU: [batch_size * num_node, timestep, in_channels]
        node_features = node_features.permute(0, 2, 1, 3).reshape(batch_size * num_node, timestep, in_channels)
        node_features, _ = self.gru1(node_features)
        node_features = node_features.reshape(batch_size * num_node, timestep, 2, -1)
        node_features = self.layer_norm1(node_features)
        node_features = node_features[:, :, 0, :] # insterested in the backward direction
        # Reshape back: [batch_size, num_node, timestep, out_channels]
        node_features = node_features.reshape(batch_size, num_node, timestep, -1)

        embeddings_over_time = []
        for t in range(timestep):

            edge_index = dynamic_edge_list[t]
            x = node_features[:, :, t, :].reshape(batch_size * num_node, -1) # Get node features at time t: [batch_size, num_node, in_channels]

            # edge_index = self.process_hyperedges_incidence(H)
            x = self.gnn(x, edge_index)
            # Reshape x to [batch_size, num_node, hidden_channels]
            x = x.reshape(batch_size, num_node, -1)
            embeddings_over_time.append(x)

        # Stack embeddings over time: [batch_size, timestep, num_node, hidden_channels]
        embeddings_over_time = torch.stack(embeddings_over_time, dim=1)

        # Pass through the second GRU
        #embeddings_over_time = embeddings_over_time.reshape(batch_size * num_node, timestep, -1)
        output = self._final_conv(embeddings_over_time)
        output = output[:, :, :, -1] # (b,c_out*T,N) for example (32, 12, 307)
        output = output.permute(0, 2, 1) # (b,T,N)-> (b,N,T)
        
        return self.sigmoid(output) #(b,N,T) for exmaple (32, 307,12)

