import torch

import torch.nn as nn
import torch.functional as F
import numpy as np


from typing import *

class AE(nn.Module):
    def __init__(self, input_features:int, hidden_features:int=4, window:int=1):
        # Encoder
        super().__init__()
        self._project = nn.Linear(in_features=window*input_features, out_features=hidden_features)
        self._fc = nn.Linear(in_features=hidden_features, out_features = hidden_features)


        # Decoder
        self._project_inv = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self._decode = nn.Linear(in_features=hidden_features, out_features=input_features)


        self._act = nn.ReLU()

    def encoder(self, x):
        x = x.view(x.shape[0], -1)
        x = self._project(x)
        x = self._act(x)
        x = self._fc(x)

        return x

    def decoder(self, x):
        x = self._project_inv(x)
        x = self._act(x)
        x = self._decode(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x