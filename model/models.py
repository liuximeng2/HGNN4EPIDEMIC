#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from model.layers import *

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax

import numpy as np

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.0):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))
        return x
    
class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spatial_out_channels):
        super(STConvBlock, self).__init__()
        self.temporal1 = GRU(in_channels, out_channels)
        self.spatial = HGNN(out_channels, spatial_out_channels, dropout=0.0)
        self.temporal2 = GRU(spatial_out_channels, out_channels)
    
    def forward(self, x, G):
        x = self.temporal1(x)
        #print(f'Size after first temporal layer: {x.size()}')
        x = self.spatial(x, G)
        #print(f'Size after first spatial layer: {x.size()}')
        x = self.temporal2(x)
        #print(f'Size after second temporal layer: {x.size()}')
        return x
    
class SDSTGCN(nn.Module):
    def __init__(self, N, in_channels, out_channels, num_blocks, kernel_size, spatial_out_channels):
        super(SDSTGCN, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(STConvBlock(in_channels, out_channels, kernel_size, spatial_out_channels))
        
        for _ in range(num_blocks - 1):
            self.blocks.append(STConvBlock(out_channels, out_channels, kernel_size, spatial_out_channels))
        
        self.final_temporal = nn.GRU(out_channels, out_channels, num_layers=1, batch_first = True)
        self.fc = nn.Linear(out_channels, 1)
    
    def forward(self, x, G):
        for block in self.blocks:
            x = block(x, G)
        
        x, _ = self.final_temporal(x)
        x = self.fc(x)
        #return F.softmax(x, dim = 1)
        return torch.sigmoid(x)

class THGNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, mlp_layer, num_node, timesteps):
        super(THGNN, self).__init__()
        self.num_node = num_node
        self.timesteps = timesteps
        self.out_channels = out_channels
        self.temporal1 = GRU(in_channels, out_channels)
        self.spatial = HGNN(timesteps * out_channels, hidden_size, dropout=0.0)
        self.fc = MLP(hidden_size, hidden_size, 1, num_layers=mlp_layer)
        
    def forward(self, x, G):
        x = self.temporal1(x)
        #print(f'Size after first temporal layer: {x.size()}')
        x = x.reshape(-1, self.num_node, self.timesteps * self.out_channels)
        #print(f'Size after first temporal layer: {x.size()}')
        x = self.spatial(x, G)
        #print(f'Size after first spatial layer: {x.size()}')
        x = self.fc(x)
        #print(f'Size after second temporal layer: {x.size()}')
        return torch.sigmoid(x)
    
class TMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, mlp_layer, num_node, timesteps):
        super(TMLP, self).__init__()
        self.num_node = num_node
        self.timesteps = timesteps
        self.out_channels = out_channels
        self.temporal1 = GRU(in_channels, out_channels)
        self.spatial = nn.Linear(timesteps * out_channels, hidden_size)
        self.fc = MLP(hidden_size, hidden_size, 1, num_layers=mlp_layer)
        
    def forward(self, x, G):
        x = self.temporal1(x)
        x = x.reshape(-1, self.num_node, self.timesteps * self.out_channels)
        x = F.relu(self.spatial(x))
        x = self.fc(x)
        return torch.sigmoid(x)