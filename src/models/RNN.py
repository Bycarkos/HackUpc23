import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from collections import Counter
import time
import os
import copy


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob=0., rnn_cell='RNN'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn_cell = rnn_cell
        if self.rnn_cell == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)
        elif self.rnn_cell == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        else:
            print('Using default Simple RNN cells')
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h, c=None):
        emb = x#self.embedding(x)
        if self.rnn_cell == 'LSTM':  # in LSTM we have a cell state and a hidden state
            out, (h, c) = self.rnn(emb, (h, c))
        else:  # in GRU and RNN we only have one hidden state
            out, h = self.rnn(emb, h)
        out = self.fc(out)
        return out, h, c

    def init_hidden(self, batch_size):
        " Initialize the hidden state of the RNN to zeros"
        weight = next(self.parameters()).data
        if self.rnn_cell == 'LSTM':  # in LSTM we have a cell state and a hidden state
            return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda"), weight.new(self.n_layers,batch_size,self.hidden_dim).zero_().to("cuda")
        else:  # in GRU and RNN we only have a hidden state
            return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda"), None

