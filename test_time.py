import torch
import numpy as np


def get_numpy():
    l = []
    for _ in range(10):
        l.append(np.zeros((100, 100)))


def get_torch():
    l = []
    for _ in range(10):
        l.append(torch.zeros((100, 100)))


if __name__ == "__main__":
    x = get_numpy()
    y = get_torch()
