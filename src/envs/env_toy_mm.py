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
    dynamic_mode = "baseline",
    fill_mode = "independent"
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
        self.state_mode = state_mode
        self.dynamics_mode = dynamic_mode

        raw = pd.read_csv(csv_path)

        self.data = build_market_features(raw)
        if state_mode == "simple":
            self.state_columns = ["mid", "spread", "imbalance"]

        elif state_mode == "engineered":
            self.state_columns = [
        "mid", "spread", "imbalance", "microprice",
        "return_1", "ma_10", "ma_20", "rsi_14"
    ]

        elif state_mode == "article_like":
            self.state_columns = [
        "rsi_14", "imbalance", "microprice",
        "ma_10", "ma_15", "ma_30"
    ]
    # Dimension d'état (+ inventory)
        self.state_dim = len(self.state_columns) + 1


    # Vérification de cohérence
        for col in self.state_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing column in data: {col}")




        self.rng = np.random.default_rng(seed)

    # paramètres de trading
        self.p_fill_base = p_fill_base
        self.eta_inv = eta_inv
        self.inv_max = inv_max
        self.inv_min = inv_min
        self.phi_as = phi_as
        self.fill_mode = fill_mode

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
    def _compute_hawkes_fill_probs(self, delta_bid, delta_ask, q_bid, q_ask, row, k = 10): 
        """
        Probabilités de fill bid / ask basées sur des intensités de Hawkes discrétisées.

        Nécessite d'initialiser dans __init__ ou reset :

            self.lambda_bid
            self.lambda_ask

        Paramètres conseillés à stocker aussi :

            self.hawkes_mu_bid
            self.hawkes_mu_ask
            self.hawkes_beta
            self.hawkes_alpha_bb
           self.hawkes_alpha_ba
            self.hawkes_alpha_aa
            self.hawkes_alpha_ab

        Returns
        -------
        p_bid_fill, p_ask_fill : float
        """

        # =========================
        # 1. Paramètres (fallback si absents)
        # =========================
        mu_bid = getattr(self, "hawkes_mu_bid", self.p_fill_base)
        mu_ask = getattr(self, "hawkes_mu_ask", self.p_fill_base)

        beta = getattr(self, "hawkes_beta", 0.20)

        alpha_bb = getattr(self, "hawkes_alpha_bb", 0.15)  # bid excite bid
        alpha_ba = getattr(self, "hawkes_alpha_ba", 0.05)  # ask excite bid

        alpha_aa = getattr(self, "hawkes_alpha_aa", 0.15)  # ask excite ask
        alpha_ab = getattr(self, "hawkes_alpha_ab", 0.05)  # bid excite ask

        # =========================
        # 2. Initialisation sécurité
        # =========================
        if not hasattr(self, "lambda_bid"):
            self.lambda_bid = mu_bid

        if not hasattr(self, "lambda_ask"):
            self.lambda_ask = mu_ask

        # =========================
        # 3. Décroissance vers la moyenne
        # =========================
        self.lambda_bid = mu_bid + (self.lambda_bid - mu_bid) * np.exp(-beta)
        self.lambda_ask = mu_ask + (self.lambda_ask - mu_ask) * np.exp(-beta)

        # =========================
        # 4. Effet quote / size
        # =========================
        quote_bid = np.exp(-k * delta_bid) / (1.0 + q_bid)
        quote_ask = np.exp(-k * delta_ask) / (1.0 + q_ask)

        lam_bid_eff = self.lambda_bid * quote_bid
        lam_ask_eff = self.lambda_ask * quote_ask

        # =========================
        # 5. Intensité -> probabilité
        # =========================
        p_bid_fill = float(np.clip(1.0 - np.exp(-lam_bid_eff), 0.0, 1.0))
        p_ask_fill = float(np.clip(1.0 - np.exp(-lam_ask_eff), 0.0, 1.0))

        # =========================
        # 6. Tirages fictifs pour mise à jour Hawkes
        # (événements de marché observés)
        # =========================
        n_bid = 1 if self.rng.random() < p_bid_fill else 0
        n_ask = 1 if self.rng.random() < p_ask_fill else 0

        # =========================
        # 7. Auto / cross excitation
        # =========================
        self.lambda_bid += alpha_bb * n_bid + alpha_ba * n_ask
        self.lambda_ask += alpha_aa * n_ask + alpha_ab * n_bid

        # garde-fou
        self.lambda_bid = max(1e-8, self.lambda_bid)
        self.lambda_ask = max(1e-8, self.lambda_ask)

        return p_bid_fill, p_ask_fill

    def step(self, action, k=10, impact_coeff=0.01):
        """
        Effectue une transition selon le mode dynamique choisi.

        En mode "baseline", le prix reste exogène.
        En mode "impact", les exécutions de l'agent déplacent le mid-price.

        Parameters
        ----------
        action : array-like
            Action scalaire ou vectorielle.
        k : float
            Sensibilité des probabilités de fill aux spreads.
        impact_coeff : float
            Intensité de l'impact sur le mid en mode "impact".
        fill_mode : str
            Mode de génération des fills :
            - "exclusive"   : au plus un fill par pas
            - "independent" : fills bid/ask tirés séparément
            - "hawkes"      : probabilités issues d'un moteur Hawkes,
                              puis tirages séparés bid/ask

        Returns
        -------
        state : np.ndarray
            Nouvel état.
        reward : float
            Reward instantané.
        done : bool
            Fin d'épisode.
        """
        row = self.data.iloc[self.t]
        fill_mode = self.fill_mode

        delta_bid, delta_ask, q_bid, q_ask = self._parse_action(action)

        # Tailles entières minimales
        q_bid = max(1, int(round(q_bid)))
        q_ask = max(1, int(round(q_ask)))

        # Quotes
        p_bid = self.mid - delta_bid
        p_ask = self.mid + delta_ask

        # =========================
        # Probabilités de fill
        # =========================
        if fill_mode == "hawkes":
            p_bid_fill, p_ask_fill = self._compute_hawkes_fill_probs(
                delta_bid, delta_ask, q_bid, q_ask, row, k=k
            )

        else:
            if self.dynamics_mode == "baseline":
                base_bid = self.p_fill_base * np.exp(-k * delta_bid)
                base_ask = self.p_fill_base * np.exp(-k * delta_ask)

                p_bid_fill = float(np.clip(base_bid / (1.0 + q_bid), 0.0, 1.0))
                p_ask_fill = float(np.clip(base_ask / (1.0 + q_ask), 0.0, 1.0))

            elif self.dynamics_mode == "impact":
                p_bid_fill, p_ask_fill = self._compute_fill_probs(
                    delta_bid, delta_ask, q_bid, q_ask, row, k=k
                )

            else:
                raise ValueError(f"Unknown dynamics_mode: {self.dynamics_mode}")

        trade_sign = 0
        did_bid = False
        did_ask = False

        # =========================
        # Gestion des fills
        # =========================
        if fill_mode == "exclusive":
            u = self.rng.random()

            if (
                u < p_bid_fill
                and self.inventory + q_bid <= self.inv_max
            ):
                self.inventory += q_bid
                self.cash -= q_bid * p_bid
                self.nb_trades += 1
                trade_sign += q_bid
                did_bid = True

            elif (
                u < p_bid_fill + p_ask_fill
                and self.inventory - q_ask >= self.inv_min
            ):
                self.inventory -= q_ask
                self.cash += q_ask * p_ask
                self.nb_trades += 1
                trade_sign -= q_ask
                did_ask = True

        elif fill_mode in ("independent", "hawkes"):
            if (
                self.inventory + q_bid <= self.inv_max
                and self.rng.random() < p_bid_fill
            ):
                self.inventory += q_bid
                self.cash -= q_bid * p_bid
                self.nb_trades += 1
                trade_sign += q_bid
                did_bid = True

            if (
                self.inventory - q_ask >= self.inv_min
                and self.rng.random() < p_ask_fill
            ):
                self.inventory -= q_ask
                self.cash += q_ask * p_ask
                self.nb_trades += 1
                trade_sign -= q_ask
                did_ask = True

        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

        # Avance temporelle
        self.t += 1

        done_data = (self.t >= len(self.data) - 1)

        done_horizon = False
        if self.max_steps is not None:
            done_horizon = (self.t - self.t0 >= self.max_steps)

        done = done_data or done_horizon

        # Mid suivant
        mid_base_next = float(self.data.iloc[self.t]["mid"]) if not done else self.mid

        if self.dynamics_mode == "baseline":
            self.mid = mid_base_next

        elif self.dynamics_mode == "impact":
            self.mid = mid_base_next + impact_coeff * trade_sign

        # Reward mark-to-market
        mtm = self.cash + self.inventory * self.mid
        reward = mtm - self.prev_mtm

        inv_penalty = self.eta_inv * self.inventory**2
        reward -= inv_penalty
        self.penalty_sum += inv_penalty

        self.inventory_path.append(self.inventory)
        self.prev_mtm = mtm

        return self._state(), float(reward), done
    

    def _parse_action(self, action):
        """
        prends l'action et la sépare de manière a pouvoir l'utiliser 

        Paramètre
        ---------
        action : tensor(T,action_dim)
            action prise dans la politique actuelle 

        Retourne 
        --------
        d_bid, d_ask, q_bid, q_ask : les quotes d et leur volume q associé. 
        """
  
        if isinstance(action, (float, int)):
            action = np.array([action], dtype=float)

        elif hasattr(action, "detach"):  # torch.Tensor
            action = action.detach().cpu().numpy()

        action = np.asarray(action, dtype=float).reshape(-1)

        # Cas 1D (ancien modèle)
        if action.shape[0] == 1:
            delta = float(action[0])
            return delta, delta, 1.0, 1.0

        # Cas 4D (nouveau modèle)
        if action.shape[0] == 4:
            d_bid, d_ask, q_bid, q_ask = action
            return float(d_bid), float(d_ask), float(q_bid), float(q_ask)

        raise ValueError(f"Action de dimension invalide : {action.shape}")
        
    def _compute_fill_probs(self, delta_bid, delta_ask, q_bid, q_ask ,row, k = 100): 
        """
    Calcule les probabilités de fill côté bid et ask.

    Parameters
    ----------
    delta_bid : float
    delta_ask : float
    q_bid : float
    q_ask : float
    row : pd.Series
        Ligne courante des données de marché
    k : float
        Sensibilité des fills au spread

    Returns
    -------
    p_bid_fill : float
    p_ask_fill : float
    """
        bv = float(row["bid_vol"])
        av = float(row["ask_vol"])
        imb = (bv - av)/(bv + av + 1e-12)
        
        base_bid = self.p_fill_base * np.exp(-k * delta_bid)
        base_ask = self.p_fill_base * np.exp(-k * delta_ask)

        # asymétrie via imbalance
        p_bid_fill = base_bid * (1.0 - imb)
        p_ask_fill = base_ask * (1.0 + imb)

        # effet taille simple : plus la quantité est grande, plus c'est difficile à exécuter
        size_bid_penalty = 1.0 / (1.0 + q_bid)
        size_ask_penalty = 1.0 / (1.0 + q_ask)

        p_bid_fill *= size_bid_penalty
        p_ask_fill *= size_ask_penalty

        p_bid_fill = float(np.clip(p_bid_fill, 0.0, 1.0))
        p_ask_fill = float(np.clip(p_ask_fill, 0.0, 1.0))

        return p_bid_fill, p_ask_fill
    

        