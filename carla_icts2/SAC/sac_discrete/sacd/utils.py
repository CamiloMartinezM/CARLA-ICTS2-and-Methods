from collections import deque

import numpy as np
import torch


def update_params(optim, loss, retain_graph=False, max_grad_norm=2, model=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if model is not None:
        print("Norming")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optim.step()


def compute_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:
    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
