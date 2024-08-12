def expand_dim(state):
    expand_dim = len(state.shape) == 1
    if expand_dim:
        state = state.unsqueeze(0) 
    return state,expand_dim

def narrow_dim(state,expand_dim):
    if expand_dim:
        state = input.squeeze(0)
    return state
