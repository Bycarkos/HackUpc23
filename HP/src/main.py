import os.path

import torch


import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


from torch.utils.data import DataLoader ,Dataset, TensorDataset

from utils.utils import *
from models.AE import *
from models.mlp import *
from models.RNN import *
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


## TODO Code just prepare for one product, easy extensible
def generate_lstm_pipeline(filepath:Path)-> pd.DataFrame:
    window = 1
    collector = Collector()
    df = read(path=filepath)
    df = df.drop(["id", "date", "year_week"], axis=1)
    X = df.iloc[:, :-1]; y=df.iloc[:, -1]

    train_loader, test_loader = collector.collect_data_by_window(x=X, window=window)
    input_dim = len(train_loader.keys())
    hidden_dim = 64
    output_dim = 1
    n_layers = 2
    drop_prob = 0.2
    num_epochs = 200
    batch_size= 64



    dataset_x = np.array(train_loader[6909]["x"])
    dataset_y = y[:len(dataset_x)].values
    embedding_dim = dataset_x[0].shape[1]
    look_up_table = torch.randn(input_dim, embedding_dim)

    train_data = TensorDataset(torch.from_numpy(dataset_x), torch.from_numpy(dataset_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    all_losses = {}
    for cell_type in ['GRU','LSTM','RNN']:
        model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob, rnn_cell=cell_type).to("cuda")
        all_losses[cell_type] = train_recurrent(train_loader, model, look_up_table, batch_size, num_epochs)
    # Visualize the loss evolution during training
    plt.figure()
    for k in all_losses.keys():
      plt.plot(all_losses[k],label=k)
    plt.legend()
    plt.show()



def main(args):

    print(f"pretrain_model_save: {args.pretrain_model_save}")
    print(f"save_embeddings: {args.save_embeddings}")
    print(f"train_type: {args.train_type}")

    if args.train_type == "lstm":
        generate_lstm_pipeline("Datasets/new_df.csv")



    # processament de les dades
    df = read(path="Datasets/new_df.csv")

    # codify cols=["prod_category", "segment", "specs", "display_size"] i emmagatzemar sequencialitat
    print(df.columns)

    #df = codify(df, cols=["specs", "display_size"])

    # netejar dades innecessaries
    df = df.drop(["id", "date", "year_week"], axis=1)

    #df = codify(df, cols=["setmana"])
    investment = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    df.iloc[:, -1] = min_max_scaler(df.iloc[:, -1].values)


    ## setup
    window = 1
    collector = Collector()
    train_loader, test_loader = collector.collect_data_by_window(x=df, window=window)
    x,y = generate_batch_from_seq(train_loader)
    x_test, y_test = generate_batch_from_seq(test_loader)

    ## Pretrain the models
    path_pretrain = os.path.join("checkpoints/", "pretrained_AE.pt")
    if os.path.exists(path_pretrain):
        model_embeddings = AE(input_features=y[0].shape[1], window=window).cuda()
        model_embeddings.load_state_dict(torch.load(path_pretrain))

    else:
        loss_list, model_embeddings = pretrain_embeddings(x=x, y=y)

    # Savaing pretrained model
    if args.pretrain_model_save == True:
        save_model(model_embeddings.state_dict, "checkpoints/", "pretrained_AE.pt")


    ## preparing the data
    embedding_dict = prepare_embedding(train_loader, test_loader, investment.values, model_embeddings)

    # Saving csv in order to get the new embeddings
    if args.save_embeddings:
        save_dictionary(embedding_dict, "Datasets/reconstructed.pkl")

    ### STARITNG THE MLP part

    embeddings = torch.tensor(np.array(embedding_dict["embedding"]))
    gt = torch.tensor(np.array(embedding_dict["gt"]))

    train_data, train_gt, test_data, test_gt = train_test_split(embeddings, gt)
    dataloader = generate_batch(train_data, train_gt)
    model_mlp = SimpleRegressor().cuda()
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loss_test = train_mlp(model_mlp, dataloader, optimizer, criterion, 200)


    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descripción del programa")
    parser.add_argument("--pretrain_model_save", type=bool, default=True, help="Descripción de pretrain_model_save")
    parser.add_argument("--save_embeddings", type=bool, default=True, help="Descripción de save_embeddings")
    parser.add_argument("--train_type", type=str, default="lstm", help="Descripción de train_type")

    args = parser.parse_args()
    main(args)