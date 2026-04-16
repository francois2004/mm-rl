from src.envs.env_toy_mm import MMSimulator
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

def inv_bin(inv, I_max=50, I_min=-50):
    inv = int(np.clip(inv, I_min, I_max))

    if inv == 0:
        return 6

    a = abs(inv)

    if a <= 2:
        offset = 1
    elif a <= 4:
        offset = 2
    elif a <= 8:
        offset = 3
    elif a <= 12:
        offset = 4
    elif a <= 25:
        offset = 5
    else:
        offset = 6

    if inv > 0:
        return 6 + offset
    else:
        return 6 - offset

def state_index(state, I_max = 50,I_min = -50):
    oi_idx = oi_bin(state[2])
    inv_idx = inv_bin(state[3], I_max, I_min)
    n_inv_bins = (I_max - I_min)//13 + 1
    s_idx = oi_idx * n_inv_bins + inv_idx
    return int(s_idx)

