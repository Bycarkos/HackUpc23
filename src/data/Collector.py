import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from typing import *



class Collector():

    def __init__(self):
        pass

    def collect_data_by_window(self, x:pd.DataFrame, window:int=1, shift=-1) -> Tuple[dict, dict]:
        unique_id = x["product_number"].unique()

        train_loader = {pid:{"x":[], "y":[]} for pid in unique_id}
        test_loader = {pid:{"x":[], "y":[]} for pid in unique_id}

        for pid in unique_id:
            x_pid = x[x["product_number"] == pid].values
            x_pid_train = x_pid[:int(x_pid.shape[0] * 0.8), 1: ]
            x_pid_test = x_pid[int(x_pid.shape[0] * 0.8):, 1: ]


            for i in range(x_pid_train.shape[0]):
                train_loader[pid]["x"].append(x_pid_train[:window, :])
                train_loader[pid]["y"].append(x_pid_train[window,:])
                x_pid_train = np.roll(x_pid_train, shift=-shift, axis=0)

            for i in range(x_pid_test.shape[0]):
                test_loader[pid]["x"].append(x_pid_test[:window, :])
                test_loader[pid]["y"].append(x_pid_test[window, :])
                x_pid_test = np.roll(x_pid_test, shift=-shift, axis=0)

        return train_loader, test_loader

    def collect_product_data(self):
        pass

    def make_data_pipeline(self):
        pass

def read(path:str="Datasets/new_df.csv", **kwargs):
    return pd.read_csv(path, **kwargs)


if __name__ == "__main__":
    collector = Collector()
    x = read()
    collector.collect_data_by_window(x)

