import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional

# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

#Temporal Layers
class GRU(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers = 1, dropout = 0.0):
        super(GRU, self).__init__()
        #self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.gru = nn.GRU(in_channels, out_channels, num_layers = num_layers, batch_first = True, dropout = dropout)
    
    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        return x


# HGNN Layers
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper
    """

    def __init__(self, in_channels, out_channels, symdegnorm=False, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.symdegnorm = symdegnorm

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = torch.matmul(x, self.weight)
        alpha = None
        if self.use_attention:
            assert num_edges <= num_edges
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.symdegnorm:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                            hyperedge_index[1], dim=0, dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0

            self.flow = 'source_to_target'
            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                                 size=(num_edges, num_nodes))
        else:  # this correspond to HGNN
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D**(0.5)
            D[D == float("inf")] = 0

            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                            hyperedge_index[1], dim=0, dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0

            x = D.unsqueeze(-1)*x
            self.flow = 'source_to_target'
            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                                 size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers = 1, dropout = 0.0):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
    
    def forward(self, x):

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x