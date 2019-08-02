'''
Methods for recording and plotting layer rotation curves
'''

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np


def get_weights(model):
    weights = dict()
    for name, value in model.named_parameters():
        if name.endswith(".weight"):
            weights[name] = value.detach().cpu().numpy()
    return weights


def compute_cosine_distances(model, initial_weights):
    current = get_weights(model)
    cos_dist = dict()
    for name, value in current.items():
        init = initial_weights[name]
        cos_dist[name] = 1 - np.average(init*value)/(
            np.sqrt(np.average(np.square(init)) *
                    np.average(np.square(value)))
        )
    return cos_dist
