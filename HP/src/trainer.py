import pandas as pd
import torch

import torch.nn as nn

import numpy as np

def train_embedding(model, loader, optimizer, criterion, device:str="cuda"):
    loss = 0
    model.train()
    for batch_features, batch_targets in zip(loader[0], loader[1]):
        # load it to the active device
        batch_features = torch.tensor(batch_features).float().to(
            device)
        batch_targets = torch.tensor(batch_targets).float().to(device)
        optimizer.zero_grad()

        # compute reconstructions
        outputs = model(batch_features)

        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_targets)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(loader)
    # print("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, epochs, loss))

    return loss


def prepare_embedding(train_loader:dict, test_loader:dict, ground:np.ndarray, model):
    print(ground)
    new_df = {"embedding": [], "gt":ground}

    index = []
    for k in train_loader.keys():
        train_loader[k]["x"].extend(test_loader[k]["x"])
        train_loader[k]["y"].extend(test_loader[k]["y"])



    for k, value in train_loader.items():
        for embedding, target in zip(value["x"], value["y"]):
            embedding = torch.tensor(embedding).float().cuda()
            with torch.no_grad():
                outputs = model.encoder(embedding).cpu().numpy()
            index.append(k)
            new_df["embedding"].append(outputs[0])

    new_df["product_id"] = index
    new_df = pd.DataFrame.from_dict(new_df)
    return new_df
