import pandas as pd
import category_encoders as ce
import numpy as np

import os
import torch
import random
import pickle


from pathlib import Path
from torch.utils.data import DataLoader ,Dataset, random_split, TensorDataset



def read(path:str="Datasets/new_df.csv", **kwargs):
    return pd.read_csv(path, kwargs)

def save_csv(df, file_path, sep=";"):
    df.to_csv(file_path, sep=sep, index=False)


import math


def compute_RMSE(y_true:torch.Tensor, y_pred:torch.Tensor):
    if len(y_true) != len(y_pred):
        raise ValueError("Missmatch shapes.")

    n = (y_true.shape[0])
    suma_cuadrados = torch.sum((y_true - y_pred) ** 2, dim=-1)
    mse = suma_cuadrados / n
    rmse = mse**0.5

    return rmse

def codify(df:pd.DataFrame, cols:list):
    encoder = ce.OneHotEncoder(cols=cols)

    return encoder.fit_transform(df)

def min_max_scaler(array):
    min_val = min(array)
    max_val = max(array)
    scaled_array = [(x - min_val) / (max_val - min_val) for x in array]

    return scaled_array


def save_model(model_state_dict: object, path:Path, filename:str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model_state_dict(), os.path.join(path, filename))



def generate_batch_from_seq(loader: dict, batch_size: int = 64):
    batched_x = []
    batched_y = []

    for pid, load in loader.items():
        batched_x.extend(load["x"])
        batched_y.extend(load["y"])

    return np.array_split(batched_x, batch_size), np.array_split(batched_y, batch_size)

def generate_batch(embeddings: torch.Tensor, gt:torch.Tensor, batch_size: int = 64):
    dataset = torch.utils.data.TensorDataset(embeddings, gt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_test_split(embeddings:torch.Tensor, gt:torch.Tensor ,prop_train:float=0.8):
    idx = torch.randperm(embeddings.shape[0])

    random.shuffle(embeddings)
    train_size = int(embeddings.shape[0]* prop_train)
    train_data = embeddings[idx][:train_size]
    test_data = embeddings[idx][train_size:]

    train_gt = gt[idx][:train_size]
    test_gt = gt[idx][train_size:]

    return train_data, train_gt,  test_data, test_gt

import pickle

def save_dictionary(dictionary, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(dictionary, file)
        print(f"The dictionary has been successfully saved to '{filename}'.")
    except IOError as e:
        print(f"An error occurred while saving the dictionary: {e}")
