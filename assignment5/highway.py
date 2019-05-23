#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, embed_size, dropout_rate=0.3):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        self.projection = nn.Linear(embed_size, embed_size)
        self.gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        X_projection = F.relu(self.projection(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = torch.mul(X_projection, X_gate) + \
                    torch.mul(X_conv_out, 1 - X_gate)
        return self.dropout(X_highway)

### END YOUR CODE 

