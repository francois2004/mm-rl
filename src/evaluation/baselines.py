"""
Implémentation de la stratégie d'Avellaneda Stoikov comme baseline analytique
"""
import numpy as np


class AvellanedaStoikovPolicy:
    def __init__(
        self,
        gamma=0.1,
        k=1.5,
        sigma=0.01,
        q_bid=1.0,
        q_ask=1.0,
        min_spread=1e-4,
        max_spread=10.0,
    ):
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.q_bid = q_bid
        self.q_ask = q_ask
        self.min_spread = min_spread
        self.max_spread = max_spread

    def reservation_price(self, mid, inventory, tau):
        return mid - inventory * self.gamma * (self.sigma ** 2) * tau

    def optimal_half_spread(self, tau):
        delta = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        delta += 0.5 * self.gamma * (self.sigma ** 2) * tau
        return np.clip(delta, self.min_spread, self.max_spread)

    def bid_ask_half_spreads(self, mid, inventory, tau):
        delta_star = self.optimal_half_spread(tau)
        skew = inventory * self.gamma * (self.sigma ** 2) * tau

        delta_bid = delta_star + skew
        delta_ask = delta_star - skew

        delta_bid = np.clip(delta_bid, self.min_spread, self.max_spread)
        delta_ask = np.clip(delta_ask, self.min_spread, self.max_spread)
        return float(delta_bid), float(delta_ask)

    def get_action_4d(self, mid, inventory, tau):
        delta_bid, delta_ask = self.bid_ask_half_spreads(mid, inventory, tau)
        return np.array([delta_bid, delta_ask, self.q_bid, self.q_ask], dtype=np.float32)

    def get_action_1d(self, mid, inventory, tau, mode="symmetric"):
        delta_bid, delta_ask = self.bid_ask_half_spreads(mid, inventory, tau)

        if mode == "symmetric":
            delta = 0.5 * (delta_bid + delta_ask)
        elif mode == "bid":
            delta = delta_bid
        elif mode == "ask":
            delta = delta_ask
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return np.array([delta], dtype=np.float32)