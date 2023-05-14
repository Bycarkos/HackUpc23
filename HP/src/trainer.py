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
    return new_df



def train_mlp(model, loader, optimizer, criterion, epochs ,device:str="cuda"):
    losses = {"train": [], "val": []}  # Two lists to keep track of the evolution of our losses

    for t in range(epochs):
        # activate training mode
        model.train()

        for idx, (x,y) in enumerate(loader):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.float().cuda()
            # Feed forward to get the logits
            y_pred = model(x)  # x_train is the whole batch, so we are doing batch gradient descent
            # Compute the loss
            loss = torch.sqrt(criterion(y, y_pred.squeeze()))

            # Backward pass to compute the gradient of loss w.r.t our learnable params
            loss.backward()

            # Update params
            optimizer.step()

            # Compute the accuracy.
            losses["train"].append(loss.item())  # keep track of our training loss

        print("Training: [EPOCH]: %i, [LOSS]: %.6f" % (t, loss.item()))

    return losses  # In case we want to plot them afterwards


def train_recurrent(dataloader, model, look_up_table, batch_size, num_epochs):
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    losses = []
    for epoch in range(num_epochs):
        new_embeddings = []

        h_state, c_state = model.init_hidden(batch_size)

        for batch, (x, y) in enumerate(dataloader):
            x = x.float().to("cuda")
            y = y.float().to("cuda")
            optimizer.zero_grad()

            y_pred, h_state, c_state = model(x, h_state, c_state)  # in LSTM we have a cell state and a hidden state
            #new_embeddings.append(nemb.detach().cpu().numpy())
            loss = criterion(y_pred.view(-1), y)**0.5

            h_state = h_state.detach()
            if c_state is not None:
                c_state = c_state.detach()

            loss.backward()
            optimizer.step()

            if batch % 30 == 0:
                print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
                losses.append(loss.item())
        #look_up_table[0] = nemb
    return losses