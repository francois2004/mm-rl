"""
Codes de transformation pour enrichir l'etat
"""

import numpy as np
import pandas as pd


def build_market_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit des features de marché à partir des données brutes du toy LOB.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame contenant au moins :
        mid, bid, ask, bid_vol, ask_vol

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi avec les features de marché.
    """
    df = df_raw.copy()

    # Sécurisation des types
    for col in ["mid", "bid", "ask", "bid_vol", "ask_vol"]:
        df[col] = df[col].astype(float)

    # Spread
    df["spread"] = df["ask"] - df["bid"]

    # Order imbalance
    denom = df["bid_vol"] + df["ask_vol"] + 1e-12
    df["imbalance"] = (df["bid_vol"] - df["ask_vol"]) / denom

    # Microprice
    df["microprice"] = (
        df["ask"] * df["bid_vol"] + df["bid"] * df["ask_vol"]
    ) / denom

    # Return simple du mid
    df["return_1"] = df["mid"].diff().fillna(0.0)

    # Moyennes mobiles du mid
    df["ma_10"] = df["mid"].rolling(window = 10, min_periods = 1).mean()
    df["ma_15"] = df["mid"].rolling(window = 15, min_periods = 1).mean()
    df["ma_20"] = df["mid"].rolling(window = 20, min_periods = 1).mean()
    df["ma_30"] = df["mid"].rolling(window = 30, min_periods = 1).mean()

    # RSI(14) construit sur les variations du mid
    delta = df["mid"].diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    return df