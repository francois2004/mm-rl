import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SAVE_DIR = Path("docs/report/images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def plot_mean_path(results, key, filename):

    arr = np.stack([r[key] for r in results])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    plt.figure(figsize=(10,5))
    plt.plot(mean)
    plt.fill_between(
        np.arange(len(mean)),
        mean-std,
        mean+std,
        alpha=0.3
    )
    plt.title(key)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=200)
    plt.close()


def plot_hist_pnl(results, filename):
    pnl = [r["final_pnl"] for r in results]

    plt.figure(figsize=(8,5))
    plt.hist(pnl, bins=30)
    plt.title("Distribution Final PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / filename, dpi=200)
    plt.close()