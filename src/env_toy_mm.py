import numpy as np 
import pandas as pd 

class MMSimulator : 
    def __init__(self, csv_path, seed = 42, p_fill_base = .30, eta_inv = .00001,
                  inv_max = 50, inv_min = -50, phi_as = .02):
        self.data = pd.read_csv(csv_path)
        self.rng = np.random.default_rng(seed)

        self.p_fill_base = p_fill_base
        self.eta_inv = eta_inv

        self.inv_max = inv_max
        self.inv_min = inv_min
        self.phi_as = phi_as

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
        self.mid = float(self.data.iloc[self.t]["mid"])

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
        self.mid = float(self.data.iloc[self.t]["mid"])

        return self._state()
    
    def _state(self):
        row = self.data.iloc[self.t]
        mid = float(self.mid)
        spread = float(row["ask"] - row["bid"])
        bv = float(row["bid_vol"])
        av = float(row["ask_vol"])
        imb = (bv - av) / (bv + av + 1e-12)
        return np.array([mid, spread, imb, self.inventory], dtype=float)
    
    def step(self, delta, k = 100): 
        
        ##quotes
        p_bid = self.mid - delta
        p_ask = self.mid + delta

        #proba de fills
        p = self.p_fill_base * np.exp(-k * float(delta))
        p = float(np.clip(p, 0.0, 1.0))
    
        #tirage + fill
        u = self.rng.random()
        #pour actualiser le mid selon le trade 
        trade_side = 0
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
        #update du prochain mid sur les datas 
        mid_base_next = float(self.data.iloc[self.t]["mid"]) if not done else self.mid
        mid_next = mid_base_next + (-trade_side)*self.phi_as
        self.mid = mid_next
        mtm = self.cash + self.inventory *mid_next
        reward = mtm - self.prev_mtm

        inv_penalty = self.eta_inv * self.inventory**2
        reward = reward - inv_penalty
        self.penalty_sum += inv_penalty

        self.inventory_path.append(self.inventory)
        self.prev_mtm = mtm

        return self._state(), float(reward), done
