import numpy as np 
import pandas as pd 

class MMSimulator : 

    def __init__(self, csv_path, seed = 42, p_fill_base = .30, eta_inv = .00001, inv_max = 50, inv_min = -50):
        self.data = pd.read_csv(csv_path)
        self.rng = np.random.default_rng(seed)

        self.p_fill_base = p_fill_base
        self.eta_inv = eta_inv

        self.inv_max = inv_max
        self.inv_min = inv_min
        self.reset()

    def reset(self): 
        self.t = 0
        self.inventory = 0
        self.cash = .0
        self.prev_mtm = .0
        self.nb_trades = 0
        self.penalty_sum = 0.0
        self.inventory_path = []
        self.nb_trades = 0
        return self._state()
    
    def reset_random(self, T_max = 50): 
        self.t = self.rng.integers(0, len(self.data)-T_max -1)
        self.inventory = 0
        self.cash = .0
        self.prev_mtm = .0
        self.nb_trades = 0
        self.penalty_sum = 0.0
        self.inventory_path = []
        self.nb_trades = 0
        return self._state()
    
    def _state(self):
        row = self.data.iloc[self.t]
        mid = float(row["mid"])
        spread = float(row["ask"] - row["bid"])
        bv = float(row["bid_vol"])
        av = float(row["ask_vol"])
        imb = (bv - av) / (bv + av + 1e-12)
        return np.array([mid, spread, imb, self.inventory], dtype=float)
    
    def step(self, delta, k = 100): 
        cur = self._state()
        mid = cur[0] 
        
        ##quotes
        p_bid = mid - delta
        p_ask = mid + delta

        p = self.p_fill_base * np.exp(-k * float(delta))
        p = float(np.clip(p, 0.0, 1.0))
        u = self.rng.random()
        if self.inventory > self.inv_min: 
            if u < p/2 : 
                self.inventory -=1
                self.cash+= p_ask
                self.nb_trades+=1
        if self.inventory < self.inv_max : 
            if (p/2 <= u< p)   : 
                self.inventory +=1
                self.cash -= p_bid
                self.nb_trades += 1

        self.t +=1
        done = (self.t >= len(self.data)-1)

        mid_next = float(self.data.iloc[self.t]["mid"]) if not done else mid

        mtm = self.cash + self.inventory *mid_next
        reward = mtm - self.prev_mtm

        inv_penalty = self.eta_inv * self.inventory**2
        reward = reward - inv_penalty
        self.penalty_sum += inv_penalty
        self.inventory_path.append(self.inventory)
        self.prev_mtm = mtm

        return self._state(), float(reward),done
