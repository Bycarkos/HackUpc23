import pandas as pd
import category_encoders as ce
import numpy as np

import os
import torch

from pathlib import Path


def read(path:str="Datasets/new_df.csv", **kwargs):
    return pd.read_csv(path, kwargs)

def save_csv(df, file_path, sep=";"):
    df.to_csv(file_path, sep=sep, index=False)

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



def generate_batch(loader: dict, batch_size: int = 64):
    batched_x = []
    batched_y = []

    for pid, load in loader.items():
        batched_x.extend(load["x"])
        batched_y.extend(load["y"])

    return np.array_split(batched_x, batch_size), np.array_split(batched_y, batch_size)



