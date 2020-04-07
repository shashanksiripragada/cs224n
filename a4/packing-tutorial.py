#https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial/blob/master/pad_packed_demo.py
###https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

import torch
import torch.nn as nn
from torch import LongTensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seqs = ['long_str',  # len = 8
        'tiny',      # len = 4
        'medium']    # len = 6
vocab = ['<pad>']+sorted(set([char for seq in seqs for char in seq]))
#print(vocab)
vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
#print(vectorized_seqs)

embed = nn.Embedding(len(vocab), 4)
lstm = nn.LSTM(4, 5, batch_first=True)

seq_lengths = LongTensor(list(map(len, vectorized_seqs)))

#print(seq_lengths)

seq_tensor = Variable(torch.zeros(len(vectorized_seqs), seq_lengths.max())).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
    seq_tensor[idx, :seqlen] = LongTensor(seq)

#print(seq_tensor)

seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
seq_tensor = seq_tensor[perm_idx]
#print(seq_tensor)

embedded_seq_tensor = embed(seq_tensor)
#print(embedded_seq_tensor)
# 3x8xvocab.vocabx4  --> 3 x 8 x 4
packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
#print(packed_input.data.shape)

packed_output, (ht,ct) = lstm(packed_input)

#print(packed_output.data.shape)
output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
print(output)
print(ht[-1])