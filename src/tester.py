import torch

import torch.nn as nn


def test_embedding(model, loader, device:str="cuda"):
    loss = 0
    model.eval()
    criterion = torch.nn.MSELoss()
    tloss = []
    for batch_features, batch_targets in zip(loader[0], loader[1]):
        batch_features = torch.tensor(batch_features).float().to(
            device)  # AQUEST INDEXING ËS PER NO TENIR EN CMOPTE EL PRODUCT ID
        batch_targets = torch.tensor(batch_targets).float().to(device)

        with torch.no_grad():
            outputs = model(batch_features)

        # compute training reconstruction loss
        test_loss = criterion(outputs, batch_targets)

        # add the mini-batch training loss to epoch loss
        loss += test_loss.item()

    # compute the epoch test loss
    loss = loss / len(loader)

    # display the epoch training loss
    # print("epoch : {}/{}, Test loss = {:.6f}".format(epoch + 1, epochs, loss))

    return loss




