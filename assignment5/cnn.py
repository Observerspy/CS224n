#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv1d, MaxPool1d
from torch.nn.functional import max_pool1d

class CNN(nn.Module):
    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5 ):
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length

        self.conv = Conv1d(
            in_channels=self.char_embed_size,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            bias=True
        )
        self.maxpool_1d = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool_1d(F.relu_(x)).squeeze()
        return x

### END YOUR CODE

