import numpy as np
import torch

def expand_dim(state):
    expand_dim = len(state.shape) == 1
    if expand_dim:
        if isinstance(state, np.ndarray):
            state = np.expand_dims(state, axis=0)
        else:
            state = state.unsqueeze(0)
    return state,expand_dim

def narrow_dim(state,expand_dim):
    if expand_dim:
        if isinstance(state, np.ndarray):
            state = np.squeeze(state, axis=0)
        else:
            state = state.squeeze(0)
    return state

