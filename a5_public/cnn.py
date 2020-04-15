#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, 
                num_filters,
                kernel_size,
                char_embed_size):
        """
        Initialize char CNN module
        @param num_filters (int): No. of output channels or word_embed_size
        @param kernel_size (int): Kernel size of each filter
        @param max_word_size (int): Max word size
        @param char_embed_size (int): embedding size of each character 
        @param padding (int): padding
        """
        super(CNN, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.char_embed_size = char_embed_size 
        self.conv1d = nn.Conv1d(in_channels=char_embed_size, out_channels=num_filters, 
                                kernel_size=kernel_size, padding=1)


    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Conv1d
        @param x_reshaped (Tensor): Reshaped Tensor of char-level embedding
        @return x_conv_out (Tensor): output of forward pass
        """
        x_conv = self.conv1d(x_reshaped)
        max_word_size = x_reshaped.shape[-1]
        pool_size = max_word_size+(2*1)-self.kernel_size+1
        self.maxpool = nn.MaxPool1d(pool_size)
        x_conv_out = self.maxpool(F.relu(x_conv))
        return torch.squeeze(x_conv_out, -1)

    ### END YOUR CODE

