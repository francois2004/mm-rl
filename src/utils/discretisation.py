from src.env_toy_mm import MMSimulator
import numpy as np

def oi_bin(oi): 
    if oi < -.6 : 
        return 0
    elif oi < -.2 : 
        return 1
    elif oi < .2:
        return 2
    elif oi < .6 : 
        return 3
    else : return 4

def inv_bin(inv, I_max = 50, I_min = -50, step = 10):
    inv_clipped = np.clip(inv, I_min, I_max)
    index = (inv_clipped - I_min)//step
    return int(index)

def state_index(state, I_max = 50,I_min = -50, step = 10):
    oi_idx = oi_bin(state[2])
    inv_idx = inv_bin(state[3], I_max, I_min, step)
    n_inv_bins = (I_max - I_min)//step + 1
    s_idx = oi_idx * n_inv_bins + inv_idx
    return int(s_idx)

