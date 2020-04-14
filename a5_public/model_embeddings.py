#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.x_char_emb = nn.Embedding(num_embeddings=len(self.vocab.char2id), 
                                    embedding_dim=50, 
                                    padding_idx=self.vocab.char_pad)
        self.cnn = CNN(num_filters=self.word_embed_size,
                kernel_size=5,
                max_word_size=21,
                char_embed_size=50,
                padding=1)
        self.highway = Highway(word_embed_size=self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        x_char_emb = self.x_char_emb(input)

        # cnn requires BATCH_SIZE,char_embed_size,max_word_size
        sen_len, batch_size, max_word_length, char_embed_size = x_char_emb.shape
        cnn_shape = (sen_len*batch_size, max_word_length, char_embed_size)
        x_reshaped = x_char_emb.view(cnn_shape).transpose(1,2)
        
        x_conv_out = self.cnn(x_reshaped)
        x_highway = self.highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        x_word_emb = x_word_emb.view(sen_len, batch_size, self.word_embed_size)
        return x_word_emb

        ### END YOUR CODE

