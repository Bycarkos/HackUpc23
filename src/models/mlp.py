import torch

import torch.nn as nn
import torch.functional as F
import numpy as np


class SimpleRegressor(nn.Module):

    def __init__(self, inDim:int=4, hiddenDim:int=4, outDim:int=1):
        super().__init__()
        self.fc1 = nn.Linear(inDim, hiddenDim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hiddenDim, outDim)

        self.red_neuronal = nn.Sequential(
            nn.Linear(in_features=inDim, out_features=hiddenDim),
            nn.ReLU(),
            nn.Linear(in_features=hiddenDim, out_features=hiddenDim),
            nn.ReLU(),
            nn.Linear(in_features=hiddenDim, out_features=hiddenDim),
            nn.ReLU(),
            nn.Linear(in_features=hiddenDim, out_features=outDim)
        )
    def forward(self, x):
        return self.red_neuronal(x)
