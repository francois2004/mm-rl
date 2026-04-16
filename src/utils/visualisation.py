""" Codes de visualisation, utiles pour le Notebook de test"""

import matplotlib.pyplot as plt

def plot_lob(df, n_points=200):
    """
    Visualise la structure du LOB :
    mid, bid, ask + indication visuelle de l'imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir 'mid', 'bid', 'ask', 'bid_vol', 'ask_vol'
    n_points : int
        Nombre de points à afficher (pour lisibilité)
    """

    data = df.iloc[:n_points].copy()

    # Imbalance
    imbalance = (data["bid_vol"] - data["ask_vol"]) / (
        data["bid_vol"] + data["ask_vol"] + 1e-12
    )

    plt.figure(figsize=(10,5))

    # Courbes principales
    plt.plot(data["mid"], label="mid", linewidth=2)
    plt.plot(data["bid"], linestyle="--", label="bid")
    plt.plot(data["ask"], linestyle="--", label="ask")

    # Remplissage visuel bid/ask
    plt.fill_between(
        range(len(data)),
        data["bid"],
        data["ask"],
        alpha=0.2,
        label="spread"
    )

    # Coloration légère selon imbalance
    plt.scatter(
        range(len(data)),
        data["mid"],
        c=imbalance,
        cmap="coolwarm",
        s=20,
        label="imbalance signal"
    )

    plt.title("LOB structure (mid / bid / ask)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.colorbar(label="imbalance")
    plt.show()