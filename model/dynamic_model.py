import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from model.layers import *
from utils.hgraph_utils import *
from utils.utils import *

from torch_geometric.data import Data
from torch_geometric.transforms import LaplacianLambdaMax

class DTHGNN(nn.Module):
    def __init__(self, args):
        super(DTHGNN, self).__init__()

        self.location_aware = args.location_aware

        self.gnn = HypergraphConv(args.in_channels, args.hidden_channels, symdegnorm=True, use_attention=False, dropout=args.dropout)
        self.gnn2 = HypergraphConv(args.hidden_channels, args.hidden_channels, symdegnorm=True, use_attention=False, dropout=args.dropout, location_aware=self.location_aware)

        self._time_conv = nn.Conv2d(
            args.hidden_channels,
            args.hidden_channels,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
        )

        self._residual_conv = nn.Conv2d(
            args.in_channels, 
            args.hidden_channels, 
            kernel_size=(1, 1), 
            stride=(1, 1)
        )

        self._final_conv = nn.Conv2d(
            int(args.len_input / args.time_strides),
            args.num_for_predict,
            kernel_size=(1, args.kernal_size),
        )
        
        # Temporal Attention Aggregator for reconstruction.
        self.node_aggregator = nn.Conv2d(
            args.hidden_channels,
            args.hidden_channels,
            kernel_size=(1, args.kernal_size)
        )

        self.edge_aggregator = nn.Conv2d(
            args.hidden_channels, 
            args.hidden_channels, 
            kernel_size=(1, args.kernal_size)
        )

        decoder_input = args.hidden_channels * (args.len_input - args.kernal_size)
        # print(f"decoder_input: {decoder_input}")
        self.num_edge = args.num_hyperedge

        self.score_decoder = MLP(decoder_input, args.hidden_channels, out_channels=1, num_layers=2, dropout=args.dropout)

        # self.gru1 = nn.GRU(args.in_channels, args.hidden_channels, args.temporal_layer, batch_first=True, bidirectional=True, dropout=args.dropout)
        # self.layer_norm1 = nn.LayerNorm(args.hidden_channels)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

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

        # node_features = node_features.permute(0, 2, 1, 3).reshape(batch_size * num_node, timestep, in_channels)
        # node_features, _ = self.gru1(node_features)
        # node_features = node_features.reshape(batch_size * num_node, timestep, 2, -1)
        # node_features = self.layer_norm1(node_features)
        # node_features = node_features[:, :, 0, :] # insterested in the backward direction
        # Reshape back: [batch_size, num_node, timestep, out_channels]
        # node_features = node_features.reshape(batch_size, num_node, timestep, -1)

        node_features = node_features.permute(0, 2, 1, 3) # [batch_size, num_node, timestep, in_channels]

        embeddings_over_time = []
        edge_embeddings_over_time = []
        for t in range(timestep):

            edge_index = dynamic_edge_list[t]
            x = node_features[:, :, t, :].reshape(batch_size * num_node, -1) # Get node features at time t: [batch_size, num_node, in_channels]
            x, _ = self.gnn(x, edge_index)

            x, edge_emb = self.gnn2(x, edge_index)
            edge_emb = edge_emb.reshape(batch_size, self.num_edge, -1)
            edge_embeddings_over_time.append(edge_emb)

            # Reshape x to [batch_size, num_node, hidden_channels]
            x = x.reshape(batch_size, num_node, -1)
            embeddings_over_time.append(x)


        # [batch_size, timestep, num_node, hidden_channels]
        x = torch.stack(embeddings_over_time, dim=1)
        edge_emb = torch.stack(edge_embeddings_over_time, dim=1)

        x_pred = self._time_conv(x.permute(0, 3, 2, 1))
        node_features = self._residual_conv(node_features.permute(0, 3, 1, 2))
        x_pred = F.relu(x_pred + node_features)
        # print(f"merged shape: {x.shape}")
        indiv_logit = self._final_conv(x_pred.permute(0, 3, 2, 1))
        indiv_logit = indiv_logit[:, :, :, -1]
        indiv_logit = indiv_logit.permute(0, 2, 1) # (b, #individual, output_timestep)


        x_time_masked = x[: , :-1, :, :]
        emb_time_masked = edge_emb[:, :-1, :, :]
        aggregated_embedding = self.node_aggregator(x_time_masked.permute(0, 3, 2, 1)).reshape(batch_size, num_node, -1)
        edge_aggregated_embedding = self.edge_aggregator(emb_time_masked.permute(0, 3, 2, 1)).reshape(batch_size, self.num_edge, -1)

        pos_node_indice = dynamic_edge_list[-1][0]
        pos_hyperedge_indices = dynamic_edge_list[-1][1]
        pos_node_embedding = aggregated_embedding[:, pos_node_indice, :]
        pos_hyperedge_indices  = edge_aggregated_embedding[:, pos_hyperedge_indices, :]

        pos_scores = (pos_node_embedding * pos_hyperedge_indices)
        # print(f"pos_scores: {pos_scores.shape}")
        pos_recon_logit = self.score_decoder(pos_scores)
        # print(f"pos_scores: {pos_scores.shape}")

        num_pos = pos_node_indice.size(0) * 2 if pos_node_indice.size(0) * 2 < 10000 else pos_node_indice.size(0)
        neg_node_indices = torch.randint(0, aggregated_embedding.size(0), (num_pos,))
        neg_hyperedge_indices = torch.randint(0, edge_aggregated_embedding.size(0), (num_pos,))
        
        neg_node_emb = aggregated_embedding[:, neg_node_indices, :]        # [num_pos, hidden_size]
        neg_hyperedge_emb = edge_aggregated_embedding[:, neg_hyperedge_indices, :]  # [num_pos, hidden_size]
        
        # Compute the dot product for negative pairs.
        neg_scores = (neg_node_emb * neg_hyperedge_emb)
        neg_recon_logit = self.score_decoder(neg_scores)

        # if evalution, return the logits
        if not self.training:
            # Ensure embeddings are on CPU before computation
            aggregated_embedding = aggregated_embedding.to("cpu")
            edge_aggregated_embedding = edge_aggregated_embedding.to("cpu")

            # Ensure the model's decoder is also on CPU
            self.score_decoder.to("cpu")

            # Expand node and hyperedge embeddings
            expanded_node_embeddings = aggregated_embedding.permute(1, 0, 2)  # Shape: (num_nodes, 1, hidden_dim)
            expanded_hyperedge_embeddings = edge_aggregated_embedding.permute(0, 1, 2)  # Shape: (1, num_hyperedges, hidden_dim)

            # Print shapes for debugging
            print(f"expanded_node_embeddings: {expanded_node_embeddings.shape}")
            print(f"expanded_hyperedge_embeddings: {expanded_hyperedge_embeddings.shape}")

            # Compute element-wise product (pairwise scores)
            full_scores = expanded_node_embeddings * expanded_hyperedge_embeddings

            # Print shape for debugging
            print(f"full_scores: {full_scores.shape}")

            # Compute the reconstructed graph using the decoder
            recon_graph = self.score_decoder(full_scores.to("cpu")).squeeze().permute(1, 0)
            print(f"recon_graph: {recon_graph.shape}")

            return indiv_logit, recon_graph, pos_recon_logit, neg_recon_logit

        return indiv_logit, _, pos_recon_logit, neg_recon_logit

    
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

