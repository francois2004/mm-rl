""" Codes de visualisation, utiles pour le Notebook de test"""
import numpy as np
import matplotlib.pyplot as plt
import torch 
import time


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


def plot_episode_dynamics(env, actor=None, max_steps=500, device=None, title_prefix=""):
    """
    Exécute un épisode et trace mid, inventory et reward sur une seule figure.

    Parameters
    ----------
    env : MMSimulator
    actor : ActorNet ou None
        Si None, actions aléatoires
    max_steps : int
        Longueur maximale de l'épisode
    device : torch.device ou None
    title_prefix : str
        Préfixe pour les titres
    """
    state = env.reset_random(max_steps=max_steps)

    mids = []
    inventories = []
    rewards = []

    done = False
    t = 0

    while not done and t < max_steps:
        mids.append(env.mid)
        inventories.append(env.inventory)

        if actor is None:
            delta = np.random.uniform(0.0, 0.05)
        else:
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                delta, _, _ = actor.sample_action(s)
            delta = float(delta.squeeze().detach().cpu().numpy())

        state, reward, done = env.step(delta)
        rewards.append(reward)
        t += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(mids)
    axes[0].set_title(f"{title_prefix}Mid")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mid")

    axes[1].plot(inventories)
    axes[1].set_title(f"{title_prefix}Inventory")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Inventory")

    axes[2].plot(rewards)
    axes[2].set_title(f"{title_prefix}Reward")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Reward")

    plt.tight_layout()
    plt.show()

def run_training_experiment(
    train_fn,
    env,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    gamma=0.99,
    n_episodes=50,
    n_epochs_actor=5,
    n_epochs_critic=5,
    random_reset=True,
    max_steps=200,
    verbose=False,
    label="Training",
):
    """
    Lance un entraînement RL et trace critic loss, actor loss et episode returns.

    Parameters
    ----------
    train_fn : callable
        Fonction d'entraînement (ex: tr_lp.train_ppo, tr.train_actor_critic)
    env : MMSimulator
    actor : ActorNet
    critic : CriticNet
    actor_optimizer : torch.optim.Optimizer
    critic_optimizer : torch.optim.Optimizer
    device : torch.device
    gamma : float
    n_episodes : int
    n_epochs_actor : int
    n_epochs_critic : int
    random_reset : bool
    max_steps : int
    verbose : bool
    label : str
        Nom affiché sur les graphes

    Returns
    -------
    history : dict
        Historique de l'entraînement
    elapsed : float
        Temps total d'entraînement
    """
    t0 = time.time()

    history = train_fn(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        gamma=gamma,
        n_episodes=n_episodes,
        n_epochs_actor=n_epochs_actor,
        n_epochs_critic=n_epochs_critic,
        random_reset=random_reset,
        max_steps=max_steps,
        verbose=verbose,
    )

    elapsed = time.time() - t0
    print("Temps total :", elapsed)

    critic_loss = history["critic_loss"]
    actor_loss = history["actor_loss"]
    returns = history["episode_return"]

    print(f"\n--- Sanity checks ({label}) ---")
    print("Nb épisodes :", len(returns))
    print("Critic loss (last) :", critic_loss[-1])
    print("Actor loss  (last) :", actor_loss[-1])
    print("Return moyen :", sum(returns) / len(returns))
    print("Return min / max :", min(returns), max(returns))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(critic_loss)
    axes[0].set_title(f"Critic loss ({label})")
    axes[0].set_xlabel("Episode")

    axes[1].plot(actor_loss)
    axes[1].set_title(f"Actor loss ({label})")
    axes[1].set_xlabel("Episode")

    axes[2].plot(returns)
    axes[2].set_title(f"Episode returns ({label})")
    axes[2].set_xlabel("Episode")

    plt.tight_layout()
    plt.show()

    return history, elapsed


def analyze_policy_actions(traj,env,actor, inventory_index=-1, imbalance_index=2):
    """
    Analyse les actions apprises sur une trajectoire.

    Parameters
    ----------
    traj : dict
        Sortie de collect_trajectory
    inventory_index : int
        Indice de l'inventory dans l'état
    imbalance_index : int
        Indice de l'imbalance dans l'état
    """
    actions = traj["actions"].detach().cpu().numpy()
    states = traj["states"].detach().cpu().numpy()
    parse = np.array([env._parse_action(a)for a in actions])
    delta_bid = parse[:, 0]
    delta_ask = parse[:, 1]
    q_bid = parse[:, 2]
    q_ask = parse[:, 3]

    inventory = states[:, inventory_index]
    imbalance = states[:, imbalance_index]

    delta_diff = delta_ask - delta_bid
    q_diff = q_ask - q_bid

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    fig.suptitle(f"Policy Analysis - Action dim = {actor.action_dim} - mode = {env.dynamics_mode}")

    axes[0, 0].hist(delta_bid, bins=25)
    axes[0, 0].set_title("delta_bid")

    axes[0, 1].hist(delta_ask, bins=25)
    axes[0, 1].set_title("delta_ask")

    axes[1, 0].hist(q_bid, bins=25)
    axes[1, 0].set_title("q_bid")

    axes[1, 1].hist(q_ask, bins=25)
    axes[1, 1].set_title("q_ask")

    axes[2, 0].scatter(inventory, delta_diff, s=8, alpha=0.5)
    axes[2, 0].set_title("delta_ask - delta_bid vs inventory")
    axes[2, 0].set_xlabel("inventory")

    axes[2, 1].scatter(inventory, q_diff, s=8, alpha=0.5)
    axes[2, 1].set_title("q_ask - q_bid vs inventory")
    axes[2, 1].set_xlabel("inventory")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(imbalance, delta_diff, s=8, alpha=0.5)
    axes[0].set_title("Spread asymmetry vs imbalance")
    axes[0].set_xlabel("imbalance")

    axes[1].scatter(imbalance, q_diff, s=8, alpha=0.5)
    axes[1].set_title("Size asymmetry vs imbalance")
    axes[1].set_xlabel("imbalance")

    plt.tight_layout()
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def analyze_policy_actions(
    traj,
    env,
    actor,
    inventory_index=-1,
    imbalance_index=2,
    save=True,
    save_dir="docs/report/images",
    prefix="policy_analysis"
):
    """
    Analyse les actions apprises sur une trajectoire
    + sauvegarde automatique des figures.

    Parameters
    ----------
    traj : dict
        Sortie de collect_trajectory
    env : env
    actor : ActorNet
    inventory_index : int
    imbalance_index : int
    save : bool
    save_dir : str
    prefix : str
    """

    os.makedirs(save_dir, exist_ok=True)

    actions = _to_numpy(traj["actions"])
    states = _to_numpy(traj["states"])

    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)

    parse = np.array([env._parse_action(a) for a in actions])

    delta_bid = parse[:, 0]
    delta_ask = parse[:, 1]
    q_bid = parse[:, 2]
    q_ask = parse[:, 3]

    inventory = states[:, inventory_index]
    imbalance = states[:, imbalance_index]

    delta_diff = delta_ask - delta_bid
    q_diff = q_ask - q_bid

    mode = getattr(env, "dynamic_mode", getattr(env, "dynamics_mode", "unknown"))

    # ==================================================
    # FIGURE 1
    # ==================================================
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    fig.suptitle(
        f"Policy Analysis - action_dim={actor.action_dim} - mode={mode}"
    )

    axes[0, 0].hist(delta_bid, bins=25)
    axes[0, 0].set_title("delta_bid")

    axes[0, 1].hist(delta_ask, bins=25)
    axes[0, 1].set_title("delta_ask")

    axes[1, 0].hist(q_bid, bins=25)
    axes[1, 0].set_title("q_bid")

    axes[1, 1].hist(q_ask, bins=25)
    axes[1, 1].set_title("q_ask")

    axes[2, 0].scatter(inventory, delta_diff, s=8, alpha=0.5)
    axes[2, 0].set_title("delta_ask - delta_bid vs inventory")
    axes[2, 0].set_xlabel("inventory")

    axes[2, 1].scatter(inventory, q_diff, s=8, alpha=0.5)
    axes[2, 1].set_title("q_ask - q_bid vs inventory")
    axes[2, 1].set_xlabel("inventory")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, f"{prefix}_inventory.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Saved:", path)

    plt.show()

    # ==================================================
    # FIGURE 2
    # ==================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(imbalance, delta_diff, s=8, alpha=0.5)
    axes[0].set_title("Spread asymmetry vs imbalance")
    axes[0].set_xlabel("imbalance")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(imbalance, q_diff, s=8, alpha=0.5)
    axes[1].set_title("Size asymmetry vs imbalance")
    axes[1].set_xlabel("imbalance")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, f"{prefix}_imbalance.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("Saved:", path)

    plt.show()

def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_training_results(
    history,
    label="Training",
    save=False,
    save_dir="docs/report/images",
    filename=None,
    ma_window=50,
    dpi=180
):
    """
    Affiche les courbes d'entraînement PPO.

    Paramètres
    ----------
    history : dict
        contient critic_loss, actor_loss, episode_return
    save : bool
        sauvegarde la figure
    save_dir : str
        dossier cible
    filename : str
        nom du fichier (sans extension)
    ma_window : int
        fenêtre moyenne mobile returns
    """

    critic_loss = np.array(history["critic_loss"], dtype=float)
    actor_loss = np.array(history["actor_loss"], dtype=float)
    returns = np.array(history["episode_return"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Critic
    axes[0].plot(critic_loss, lw=1.2)
    axes[0].set_title("Critic loss")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)

    # Actor
    axes[1].plot(actor_loss, lw=1.2)
    axes[1].set_title("Actor loss")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.3)

    # Returns
    axes[2].plot(returns, alpha=0.35, lw=1.0, label="raw")

    ma = moving_average(returns, ma_window)
    if len(returns) >= ma_window:
        axes[2].plot(
            range(ma_window - 1, len(returns)),
            ma,
            lw=2,
            label=f"MA({ma_window})"
        )

    axes[2].set_title("Episode returns")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(label, fontsize=14)
    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            filename = label.lower().replace(" ", "_")

        png_path = os.path.join(save_dir, filename + ".png")
        pdf_path = os.path.join(save_dir, filename + ".pdf")

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        print("Saved:", png_path)
        print("Saved:", pdf_path)

    plt.show()
