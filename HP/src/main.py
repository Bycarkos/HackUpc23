import os.path

import torch


import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader ,Dataset

from utils.utils import *
from models.AE import *
from data.Collector import *

from trainer import *
from tester import *


def pretrain_embeddings(x:np.ndarray, y:np.ndarray,window:int=1, epochs:int=200):

    model = AE(input_features=y[0].shape[1], window=window).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = []
    loss_test = []

    for epoch in range(epochs):
        idx = torch.randperm(len(x))
        x = np.array(x, dtype="object")[idx]
        y = np.array(y,dtype="object")[idx]
        loss.append(train_embedding(model=model, loader=(x,y), optimizer=optimizer, criterion=criterion ))

    return loss, model




def main(pretrain_model_save:bool = True, save_embeddings:bool = True):

    # processament de les dades
    df = read(path="Datasets/new_df.csv")

    # codify cols=["prod_category", "segment", "specs", "display_size"] i emmagatzemar sequencialitat
    print(df.columns)

    df = codify(df, cols=["specs", "display_size"])

    # netejar dades innecessaries
    df = df.drop(["id", "date", "year_week"], axis=1)

    df = codify(df, cols=["setmana"])
    investment = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    df.iloc[:, -1] = min_max_scaler(df.iloc[:, -1].values)


    ## setup
    window = 1
    collector = Collector()
    train_loader, test_loader = collector.collect_data_by_window(x=df, window=window)
    x,y = generate_batch(train_loader)
    x_test, y_test = generate_batch(test_loader)

    ## Pretrain the models
    path_pretrain = os.path.join("checkpoints/", "pretrained_AE.pt")
    if os.path.exists(path_pretrain):
        model_embeddings = AE(input_features=y[0].shape[1], window=window).cuda()
        model_embeddings.load_state_dict(torch.load(path_pretrain))

    else:
        loss_list, model_embeddings = pretrain_embeddings(x=x, y=y)

    # Savaing pretrained model
    if pretrain_model_save == True:
        save_model(model_embeddings.state_dict, "checkpoints/", "pretrained_AE.pt")


    ## preparing the data
    embedding_csv = prepare_embedding(train_loader, test_loader, investment.values, model_embeddings)

    # Saving csv in order to get the new embeddings
    if save_embeddings:
        save_csv(embedding_csv, "Datasets/encoded.csv")

    loss_test = test_embedding(loader=(x_test,y_test), model=model_embeddings)
    print(f"The loss with the AE in level test is: {loss_test} ")


    ### STARITNG THE LSTM part


if __name__ == "__main__":
    #loss= pretrain_embeddings(x=x, y=y)
    main()