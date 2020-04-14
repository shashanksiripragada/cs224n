#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embed_size):
        """ Init Higway network
        @param word_embed_size (int): Embedding size of word, in handout, 
                                     it's e_{word} (dimensionality)
        """
        super(Highway, self).__init__()

        self.W_proj = nn.Linear(word_embed_size, word_embed_size)
        self.W_gate = nn.Linear(word_embed_size, word_embed_size) 


    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Take mini-batch of sentence of ConvNN
        @param x_conv_out (Tensor): Tensor with shape (max_sentence_length, batch_size, embed_size)
        @return x_highway (Tensor): combinded output with shape (max_sentence_length, batch_size, embed_size)
        """
        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul(1-x_gate, x_conv_out)

        return x_highway 
    ### END YOUR CODE

