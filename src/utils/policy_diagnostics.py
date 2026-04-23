import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.envs.env_toy_mm import MMSimulator


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rollout_detailed(
    env : MMSimulator,
    actor,
    device,
    max_steps=200,
    random_reset=True,
    deterministic=False,
):
    """
    Rejoue une politique sur un épisode et enregistre un maximum
    d'informations utiles pour le diagnostic.

    Retourne un dict de arrays numpy.
    """
    actor.eval()

    if random_reset and hasattr(env, "reset_random"):
        state = env.reset_random(max_steps=max_steps)
    else:
        state = env.reset()

    traj = {
        "t": [],
        "states": [],
        "actions_raw": [],
        "delta_bid": [],
        "delta_ask": [],
        "q_bid": [],
        "q_ask": [],
        "inventory": [],
        "cash": [],
        "mid": [],
        "mtm": [],
        "rewards": [],
        "cum_rewards": [],
    }

    done = False
    step = 0
    cum_reward = 0.0

    while not done and step < max_steps:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                if hasattr(actor, "act_deterministic"):
                    action = actor.act_deterministic(s)
                else:
                    # fallback : on prend la moyenne implicite si dispo
                    # sinon sample_action
                    action, _, _ = actor.sample_action(s)
            else:
                action, _, _ = actor.sample_action(s)

        action_np = _to_numpy(action).reshape(-1)
        delta_bid, delta_ask, q_bid, q_ask = env._parse_action(action_np)

        next_state, reward, done = env.step(action_np)

        mtm = env.cash + env.inventory * env.mid
        cum_reward += float(reward)

        traj["t"].append(step)
        traj["states"].append(np.array(state, dtype=float))
        traj["actions_raw"].append(np.array(action_np, dtype=float))
        traj["delta_bid"].append(float(delta_bid))
        traj["delta_ask"].append(float(delta_ask))
        traj["q_bid"].append(float(q_bid))
        traj["q_ask"].append(float(q_ask))
        traj["inventory"].append(float(env.inventory))
        traj["cash"].append(float(env.cash))
        traj["mid"].append(float(env.mid))
        traj["mtm"].append(float(mtm))
        traj["rewards"].append(float(reward))
        traj["cum_rewards"].append(float(cum_reward))

        state = next_state
        step += 1

    for k in traj:
        traj[k] = np.array(traj[k])

    return traj


def plot_policy_timeseries(
    traj,
    save=True,
    save_dir="docs/report/images",
    prefix="policy_timeseries"
):
    """
    Graphes temporels essentiels :
    - quotes
    - tailles
    - inventory
    - mid
    - mtm
    - reward cumulée
    """
    os.makedirs(save_dir, exist_ok=True)

    t = traj["t"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Quotes
    axes[0, 0].plot(t, traj["delta_bid"], label="delta_bid")
    axes[0, 0].plot(t, traj["delta_ask"], label="delta_ask")
    axes[0, 0].set_title("Quotes over time")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    # Sizes
    axes[0, 1].plot(t, traj["q_bid"], label="q_bid")
    axes[0, 1].plot(t, traj["q_ask"], label="q_ask")
    axes[0, 1].set_title("Sizes over time")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    # Inventory
    axes[1, 0].plot(t, traj["inventory"])
    axes[1, 0].set_title("Inventory")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].grid(alpha=0.25)

    # Mid
    axes[1, 1].plot(t, traj["mid"])
    axes[1, 1].set_title("Mid-price")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].grid(alpha=0.25)

    # MtM / PnL
    axes[2, 0].plot(t, traj["mtm"])
    axes[2, 0].set_title("Mark-to-market")
    axes[2, 0].set_xlabel("t")
    axes[2, 0].grid(alpha=0.25)

    # Cumulated reward
    axes[2, 1].plot(t, traj["cum_rewards"])
    axes[2, 1].set_title("Cumulated reward")
    axes[2, 1].set_xlabel("t")
    axes[2, 1].grid(alpha=0.25)

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, f"{prefix}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Saved:", path)

    plt.show()


def plot_policy_state_links(
    traj,
    env,
    inventory_index=-1,
    imbalance_index=None,
    microprice_index=None,
    save=True,
    save_dir="docs/report/images",
    prefix="policy_state_links"
):
    """
    Graphes de dépendance action/état :
    - inventory vs skew
    - inventory vs size asymmetry
    - imbalance vs quotes/sizes si dispo
    - microprice vs quotes si dispo
    """
    os.makedirs(save_dir, exist_ok=True)

    states = traj["states"]
    inventory = states[:, inventory_index]

    delta_diff = traj["delta_ask"] - traj["delta_bid"]
    q_diff = traj["q_ask"] - traj["q_bid"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(inventory, delta_diff, s=10, alpha=0.5)
    axes[0, 0].set_title("delta_ask - delta_bid vs inventory")
    axes[0, 0].set_xlabel("inventory")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(inventory, q_diff, s=10, alpha=0.5)
    axes[0, 1].set_title("q_ask - q_bid vs inventory")
    axes[0, 1].set_xlabel("inventory")
    axes[0, 1].grid(alpha=0.25)

    if imbalance_index is not None:
        imbalance = states[:, imbalance_index]

        axes[1, 0].scatter(imbalance, delta_diff, s=10, alpha=0.5)
        axes[1, 0].set_title("spread asymmetry vs imbalance")
        axes[1, 0].set_xlabel("imbalance")
        axes[1, 0].grid(alpha=0.25)
    else:
        axes[1, 0].axis("off")

    if microprice_index is not None:
        microprice = states[:, microprice_index]

        axes[1, 1].scatter(microprice, traj["delta_bid"], s=10, alpha=0.5, label="delta_bid")
        axes[1, 1].scatter(microprice, traj["delta_ask"], s=10, alpha=0.5, label="delta_ask")
        axes[1, 1].set_title("quotes vs microprice")
        axes[1, 1].set_xlabel("microprice")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.25)
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, f"{prefix}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Saved:", path)

    plt.show()


def print_rollout_summary(traj):
    """
    Petit résumé numérique utile.
    """
    print("=== Rollout summary ===")
    print(f"steps           : {len(traj['t'])}")
    print(f"final inventory : {traj['inventory'][-1]:.4f}")
    print(f"final cash      : {traj['cash'][-1]:.4f}")
    print(f"final mtm       : {traj['mtm'][-1]:.4f}")
    print(f"cum reward      : {traj['cum_rewards'][-1]:.4f}")
    print(f"mean delta_bid  : {traj['delta_bid'].mean():.4f}")
    print(f"mean delta_ask  : {traj['delta_ask'].mean():.4f}")
    print(f"mean q_bid      : {traj['q_bid'].mean():.4f}")
    print(f"mean q_ask      : {traj['q_ask'].mean():.4f}")