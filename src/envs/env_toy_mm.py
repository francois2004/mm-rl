import numpy as np 
import pandas as pd 
from src.features.market_features import build_market_features

class MMSimulator : 
    """
    Simulateur simplifié de market making basé sur des données historiques de LOB.

    Ce simulateur implémente un environnement de type MDP discret (temps discret,
    état observable, dynamique exogène) destiné à l'entraînement d'agents de
    reinforcement learning pour des stratégies de market making.

    Hypothèses principales :
    ------------------------
    - Le mid-price est exogène (issu des données), avec un léger impact simulé.
    - Les probabilités d'exécution sont modélisées de manière réduite via une loi
      exponentielle décroissante en fonction du spread (cf. Avellaneda-Stoikov).
    - Les ordres sont unitaires (±1 en inventory).
    - Le reward est basé sur une variation de mark-to-market pénalisée par l'inventory.

    Limites :
    ---------
    - Absence de dynamique endogène du carnet (pas de reconstruction du LOB).
    - Modèle de fill extrêmement simplifié (pas de file d'attente ni de priorité).
    - Impact de marché réduit à un shift affine arbitraire.

    Paramètres
    ----------
    csv_path : str
        Chemin vers les données de marché (colonnes attendues : mid, bid, ask,
        bid_vol, ask_vol).
    seed : int
        Graine pour la reproductibilité.
    p_fill_base : float
        Probabilité de fill à spread nul.
    eta_inv : float
        Coefficient de pénalité quadratique sur l'inventaire.
    inv_max, inv_min : int
        Bornes sur l'inventaire.
    phi_as : float
        Paramètre d'impact adverse (shift du mid après trade).
    """
    def __init__(
    self,
    csv_path,
    seed=42,
    p_fill_base=0.30,
    eta_inv=1e-5,
    inv_max=50,
    inv_min=-50,
    phi_as=0.02,
    state_mode="simple", 
    dynamic_mode = "baseline"
):
        """
    Simulateur de market making sur données LOB.

    Parameters
    ----------
    csv_path : str
        Chemin vers les données brutes.
    state_mode : str
        "simple" ou "engineered"
    """
        raw = pd.read_csv(csv_path)

        self.state_mode = state_mode
        self.dynamics_mode = dynamic_mode

        if state_mode == "simple":
            self.data = raw.copy()

        # Features minimales construites à la volée
            self.data["spread"] = self.data["ask"] - self.data["bid"]
            denom = self.data["bid_vol"] + self.data["ask_vol"] + 1e-12
            self.data["imbalance"] = (self.data["bid_vol"] - self.data["ask_vol"]) / denom

            self.state_columns = ["mid", "spread", "imbalance"]

        elif state_mode == "engineered":
            self.data = build_market_features(raw)

            self.state_columns = [
            "mid",
            "spread",
            "imbalance",
            "microprice",
            "return_1",
            "ma_10",
            "ma_20",
            "rsi_14",
        ]

        else:
            raise ValueError(f"Unknown state_mode: {state_mode}")

    # Vérification de cohérence
        for col in self.state_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing column in data: {col}")

    # Dimension d'état (+ inventory)
        self.state_dim = len(self.state_columns) + 1


        self.rng = np.random.default_rng(seed)

    # paramètres de trading
        self.p_fill_base = p_fill_base
        self.eta_inv = eta_inv
        self.inv_max = inv_max
        self.inv_min = inv_min
        self.phi_as = phi_as

    # gestion épisode
        self.max_steps = None
        self.t0 = None

        self.reset()

    def reset(self, max_steps = None): 
        """
        Réinitialise l'environnement au début des données.

        Initialise :
        - temps t
        - inventory
        - cash
        - mark-to-market précédent

        Returns
        -------
        np.ndarray
            État initial (mid, spread, imbalance, inventory).
        """
        self.t0 = 0
        self.t = 0
        self.max_steps = max_steps

        self.inventory = 0
        self.cash = .0
        self.prev_mtm = .0
        self.nb_trades = 0
        self.penalty_sum = 0.0
        self.inventory_path = []
        self.nb_trades = 0
        self.mid = float(self.data.iloc[self.t]["mid"])

        return self._state()
    
    def reset_random(self, max_steps=50):
        """
        Réinitialise l'environnement à un point aléatoire des données.

        Utile pour éviter le surapprentissage sur une unique trajectoire.

        Paramètres
        ----------
        T_max : int
            Horizon maximal restant pour garantir une longueur minimale d'épisode.

        Returns
        -------
        np.ndarray
            État initial.
        """
        self.max_steps = max_steps

        self.t0 = self.rng.integers(0, len(self.data) - max_steps - 1)
        self.t = self.t0

        self.inventory = 0
        self.cash = 0.0
        self.prev_mtm = 0.0
        self.nb_trades = 0
        self.penalty_sum = 0.0
        self.inventory_path = []

        self.mid = float(self.data.iloc[self.t]["mid"])

        return self._state()
    
    def _state(self):
        """
    Construit l'état courant observé par l'agent.

    Returns
    -------
    np.ndarray
        Etat de dimension state_dim
    """
        row = self.data.iloc[self.t]

        market_state = [float(row[col]) for col in self.state_columns]

        return np.array(
        market_state + [float(self.inventory)],
        dtype=float
    )
    
    def _step_baseline(self, delta, k = 100): 
        """
    Transition de l'environnement avec impact simplifié de l'agent.

    Les probabilités de fill dépendent du spread et de l'imbalance.
    Les transactions influencent directement le mid via un terme d'impact.
    L'agent agit donc à la fois sur ses exécutions et sur la dynamique du prix.

    Paramètres
    ----------
    delta : float
        Distance au mid choisie par l'agent.

    Retourne
    --------
    state : np.ndarray
        Nouvel état après transition.
    reward : float
        Variation de PnL pénalisée par l'inventaire.
    done : bool
        Indique la fin de l'épisode.
    """
        ##quotes
        p_bid = self.mid - delta
        p_ask = self.mid + delta

        #proba de fills
        p = self.p_fill_base * np.exp(-k * float(delta))
        p = float(np.clip(p, 0.0, 1.0))
    
        #tirage + fill
        u = self.rng.random()
        #pour actualiser le mid selon le trade 
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

        self.t += 1

        done_data = (self.t >= len(self.data) - 1)

        done_horizon = False
        if self.max_steps is not None:
            done_horizon = (self.t - self.t0 >= self.max_steps)

        done = done_data or done_horizon
        #update du prochain mid sur les datas 
        mid_base_next = float(self.data.iloc[self.t]["mid"]) if not done else self.mid
        mid_next = mid_base_next 
        self.mid = mid_next
        mtm = self.cash + self.inventory *mid_next
        reward = mtm - self.prev_mtm

        inv_penalty = self.eta_inv * self.inventory**2
        reward = reward - inv_penalty
        self.penalty_sum += inv_penalty

        self.inventory_path.append(self.inventory)
        self.prev_mtm = mtm

        return self._state(), float(reward), done
    
    def _step_impact(self, delta, k=100, impact_coeff=0.01):
        """
        Transition de l'environnement avec impact simplifié de l'agent.

        Les probabilités de fill dépendent du spread et de l'imbalance.
        Les transactions influencent directement le mid via un terme d'impact.
        L'agent agit donc à la fois sur ses exécutions et sur la dynamique du prix.

        Paramètres
        ----------
        delta : float
            Distance au mid choisie par l'agent.
        k : float
            Paramètre de décroissance des probabilités de fill.
        impact_coeff : float
            Intensité de l'impact des trades sur le mid-price.

        Retourne
        --------
        state : np.ndarray
            Nouvel état après transition.
        reward : float
            Variation de PnL pénalisée par l'inventaire.
        done : bool
            Indique la fin de l'épisode.
        """
        # Données de marché au temps courant
        row = self.data.iloc[self.t]
        bv = float(row["bid_vol"])
        av = float(row["ask_vol"])
        imb = (bv - av) / (bv + av + 1e-12)

        # Quotes de l'agent
        p_bid = self.mid - delta
        p_ask = self.mid + delta

        # Proba de fill de base
        base_p = self.p_fill_base * np.exp(-k * float(delta))
        base_p = float(np.clip(base_p, 0.0, 1.0))

        # Asymétrie via imbalance
        p_bid_fill = np.clip(base_p * (1.0 - imb), 0.0, 1.0)
        p_ask_fill = np.clip(base_p * (1.0 + imb), 0.0, 1.0)

        trade_sign = 0

        # Fill côté bid : l'agent achète
        if self.inventory < self.inv_max:
            if self.rng.random() < p_bid_fill:
                self.inventory += 1
                self.cash -= p_bid
                self.nb_trades += 1
                trade_sign += 1

        # Fill côté ask : l'agent vend
        if self.inventory > self.inv_min:
            if self.rng.random() < p_ask_fill:
                self.inventory -= 1
                self.cash += p_ask
                self.nb_trades += 1
                trade_sign -= 1

        # Avance temporelle
        self.t += 1

        done_data = (self.t >= len(self.data) - 1)

        done_horizon = False
        if self.max_steps is not None:
            done_horizon = (self.t - self.t0 >= self.max_steps)

        done = done_data or done_horizon

        # Mid exogène + impact des trades
        mid_base_next = float(self.data.iloc[self.t]["mid"]) if not done else self.mid
        mid_next = mid_base_next + impact_coeff * trade_sign
        self.mid = mid_next

        # Reward mark-to-market
        mtm = self.cash + self.inventory * mid_next
        reward = mtm - self.prev_mtm

        inv_penalty = self.eta_inv * self.inventory**2
        reward -= inv_penalty
        self.penalty_sum += inv_penalty

        self.inventory_path.append(self.inventory)
        self.prev_mtm = mtm

        return self._state(), float(reward), done
    
    def step(self, delta, k=100): 
        """
        Appelle la fonction de step associée au mode de dynamique initialisé dans l'env 
        """
        if self.dynamics_mode == "baseline":
            return self._step_baseline(delta, k=k)
        elif self.dynamics_mode == "impact":
            return self._step_impact(delta, k=k)
        else:
            raise ValueError(f"Unknown dynamics_mode: {self.dynamics_mode}")
        