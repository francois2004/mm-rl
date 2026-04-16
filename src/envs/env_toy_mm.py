import numpy as np 
import pandas as pd 

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
    def __init__(self, csv_path, seed = 42, p_fill_base = .30, eta_inv = .00001,
                  inv_max = 50, inv_min = -50, phi_as = .02):
        self.data = pd.read_csv(csv_path)
        self.rng = np.random.default_rng(seed)

        self.p_fill_base = p_fill_base
        self.eta_inv = eta_inv

        self.inv_max = inv_max
        self.inv_min = inv_min
        self.phi_as = phi_as

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

        État = (mid, spread, order imbalance, inventory)

        - imbalance : (bid_vol - ask_vol) / (bid_vol + ask_vol)

        Returns
        -------
        np.ndarray
            Vecteur d'état de dimension 4.
        """     
        row = self.data.iloc[self.t]
        mid = float(self.mid)
        spread = float(row["ask"] - row["bid"])
        bv = float(row["bid_vol"])
        av = float(row["ask_vol"])
        imb = (bv - av) / (bv + av + 1e-12)
        return np.array([mid, spread, imb, self.inventory], dtype=float)
    
    def step(self, delta, k = 100, trade_side =0): 
        """
        Effectue une transition de l'environnement.

        L'agent choisit un spread symétrique delta autour du mid :
            p_bid = mid - delta
            p_ask = mid + delta

        La probabilité d'exécution suit :
            p_fill = p0 * exp(-k * delta)

        Le fill est simulé par un tirage uniforme.

        Paramètres
        ----------
        delta : float
            Distance au mid (contrôle principal de l'agent).
        k : float
            Paramètre de décroissance de la probabilité de fill.
        trade_side : int
            Paramètre d'impact (±1 ou 0), censé représenter l'adverse selection.

        Returns
        -------
        state : np.ndarray
            Nouvel état.
        reward : float
            Reward instantané (variation de PnL pénalisée).
        done : bool
            Indicateur de fin d'épisode.

        Remarques critiques
        -------------------
        - Le modèle de fill ne dépend pas du carnet → irréaliste.
        - Le trade_side n'est pas endogène au fill → incohérent économiquement.
        - Le reward est du type mark-to-market, ce qui introduit du bruit
          important (problème classique en RL financier).
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
