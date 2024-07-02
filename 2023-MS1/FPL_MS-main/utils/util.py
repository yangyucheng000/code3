import os
import numpy as np



def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def save_networks(model, communication_idx):
   pass 


def save_protos(model, communication_idx):
    pass
