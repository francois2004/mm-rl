import numpy as np
import pandas as pd
import os

def generate_toy_lob_simple(N=10_000,mid0=100.0,dt=1/390,
    mu=0.0,sigma=0.02,spread=0.01,lambda_q=10.0,seed=42,
    save = True, save_dir = "data/raw"
):
    """
    Générateur de LOB simpliste (baseline).

    Parameters
    ----------
    N : int
        Nombre de points
    mid0 : float
        Prix initial
    dt : float
        Pas de temps
    mu : float
        Drift constant
    sigma : float
        Volatilité constante
    spread : float
        Spread constant
    lambda_q : float
        Intensité Poisson pour les volumes
    seed : int
        Graine aléatoire

    Returns
    -------
    pd.DataFrame
        Colonnes : mid, bid, ask, bid_vol, ask_vol
    """
    rng = np.random.default_rng(seed)

    mid = np.zeros(N)
    bid = np.zeros(N)
    ask = np.zeros(N)

    mid[0] = mid0

    for t in range(1, N):
        z = rng.normal()
        dS = mu * mid[t-1] * dt + sigma * mid[t-1] * np.sqrt(dt) * z
        mid[t] = mid[t-1] + dS

    # bid / ask
    bid = mid - spread / 2
    ask = mid + spread / 2

    # volumes iid
    bid_vol = rng.poisson(lambda_q, size=N)
    ask_vol = rng.poisson(lambda_q, size=N)

    df = pd.DataFrame({
        "mid": mid,
        "bid": bid,
        "ask": ask,
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
    })
    if save:
        os.makedirs(save_dir, exist_ok=True)

        filename = f"toy_lob_simple_seed{seed}.csv"

        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)

        print(f"[generate_toy_lob_simple] saved to {path}")


    return df




def generate_toy_lob_nonstationary(N=10_000,mid0=100.0,dt=1/390,seed=42,
    # OU drift
    mu_bar=0.0,mu0=0.0,kappa_mu=5.0,eta_mu=0.02,
    # CIR spread
    s_bar=0.01,s0=0.01,kappa_s=8.0,sigma_s=0.02,
    # GARCH volatility
    omega=1e-6,alpha=0.08,beta=0.90,sigma0=0.02,
    # volumes
    lambda_q=10.0,vol_sensitivity=8.0,
    save = True, save_dir = "data/raw"
):
    """
    Génère un toy LOB non stationnaire inspiré de l'article.

    Retourne
    -------
    pd.DataFrame
        Colonnes : mid, bid, ask, bid_vol, ask_vol
    """
    rng = np.random.default_rng(seed)

    mid = np.zeros(N)
    bid = np.zeros(N)
    ask = np.zeros(N)
    bid_vol = np.zeros(N, dtype=int)
    ask_vol = np.zeros(N, dtype=int)

    mu_t = np.zeros(N)
    spread_t = np.zeros(N)
    sigma2_t = np.zeros(N)
    eps_t = np.zeros(N)

    mid[0] = mid0
    mu_t[0] = mu0
    spread_t[0] = s0
    sigma2_t[0] = sigma0**2

    for t in range(1, N):
        # 1) Drift OU
        z_mu = rng.normal()
        mu_t[t] = (
            mu_t[t-1]
            + kappa_mu * (mu_bar - mu_t[t-1]) * dt
            + eta_mu * np.sqrt(dt) * z_mu
        )

        # 2) Spread CIR discretisé (positif)
        z_s = rng.normal()
        spread_t[t] = (
            spread_t[t-1]
            + kappa_s * (s_bar - spread_t[t-1]) * dt
            + sigma_s * np.sqrt(max(spread_t[t-1], 1e-8)) * np.sqrt(dt) * z_s
        )
        spread_t[t] = max(spread_t[t], 1e-4)

        # 3) Volatilité GARCH
        sigma2_t[t] = omega + alpha * eps_t[t-1]**2 + beta * sigma2_t[t-1]
        sigma_t = np.sqrt(max(sigma2_t[t], 1e-10))

        # 4) Rendement du mid
        z = rng.normal()
        ret = mu_t[t] * dt + sigma_t * np.sqrt(dt) * z
        eps_t[t] = ret
        mid[t] = mid[t-1] * (1.0 + ret)

        # 5) Bid / ask autour du mid
        bid[t] = mid[t] - spread_t[t] / 2.0
        ask[t] = mid[t] + spread_t[t] / 2.0

        # 6) Volumes Poisson avec légère dépendance au signe du drift
        lam_bid = lambda_q * max(0.2, 1.0 + vol_sensitivity * max(mu_t[t], 0.0))
        lam_ask = lambda_q * max(0.2, 1.0 + vol_sensitivity * max(-mu_t[t], 0.0))

        bid_vol[t] = rng.poisson(lam_bid)
        ask_vol[t] = rng.poisson(lam_ask)

    # init t=0
    bid[0] = mid[0] - spread_t[0] / 2.0
    ask[0] = mid[0] + spread_t[0] / 2.0
    bid_vol[0] = rng.poisson(lambda_q)
    ask_vol[0] = rng.poisson(lambda_q)

    df = pd.DataFrame({
        "mid": mid,
        "bid": bid,
        "ask": ask,
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
    })
    if save:
        os.makedirs(save_dir, exist_ok=True)

        filename = f"toy_lob_non_stationnary_seed{seed}.csv"

        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)

        print(f"[generate_toy_lob_simple] saved to {path}")

    return df

