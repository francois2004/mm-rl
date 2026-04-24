import os
import json
import torch

from src.envs.env_toy_mm import MMSimulator
from src.ppo.networks import ActorNet, CriticNet
from src.utils.device import get_device
from scripts.train_loop import train_ppo


def run_one_experiment(fill_mode, out_dir, csv_path, device):
    print(f"\n=== Run fill_mode = {fill_mode} ===")

    env = MMSimulator(
        csv_path=csv_path,
        seed=42,
        p_fill_base=0.30,
        eta_inv=1e-3,
        inv_max=50,
        inv_min=-50,
        phi_as=0.02,
        state_mode="article_like",
        dynamic_mode="impact",
        fill_mode=fill_mode,
    )

    actor = ActorNet(
        state_dim=env.state_dim,
        hidden_size=64,
        n_layers=3,
        action_dim=4,
        delta_min=0.0,
        delta_max=0.05,
    ).to(device)

    critic = CriticNet(
        state_dim=env.state_dim,
        hidden_size=64,
        n_layers=3,
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    history = train_ppo(
        env=env,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        device=device,
        gamma=0.99,
        lam=0.95,
        n_episodes=500,
        n_epochs_actor=10,
        n_epochs_critic=20,
        batch_size=256,
        random_reset=True,
        max_steps=256,
        verbose=True,
    )

    os.makedirs(out_dir, exist_ok=True)

    torch.save(actor.state_dict(), os.path.join(out_dir, "actor.pt"))
    torch.save(critic.state_dict(), os.path.join(out_dir, "critic.pt"))

    config = {
        "fill_mode": fill_mode,
        "csv_path": csv_path,
        "seed": 42,
        "p_fill_base": 0.30,
        "eta_inv": 1e-3,
        "inv_max": 50,
        "inv_min": -50,
        "phi_as": 0.02,
        "state_mode": "article_like",
        "dynamic_mode": "impact",
        "state_dim": env.state_dim,
        "hidden_size": 64,
        "n_layers": 3,
        "action_dim": 4,
        "delta_min": 0.0,
        "delta_max": 0.05,
        "gamma": 0.99,
        "lam": 0.95,
        "n_episodes": 500,
        "n_epochs_actor": 10,
        "n_epochs_critic": 20,
        "batch_size": 256,
        "max_steps": 256,
        "lr": 3e-4,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(
            {
                "critic_loss": [float(x) for x in history["critic_loss"]],
                "actor_loss": [float(x) for x in history["actor_loss"]],
                "episode_return": [float(x) for x in history["episode_return"]],
            },
            f,
            indent=2,
        )

    print(f"Run terminé. Résultats sauvegardés dans {out_dir}")


def main():
    device = get_device()
    print("Device:", device)

    csv_path = "data/raw/toy_lob_non_stationnary_seed42.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    fill_modes = [ "hawkes"]

    base_out_dir = "logs/nouvel_inventaire"
    os.makedirs(base_out_dir, exist_ok=True)

    for fill_mode in fill_modes:
        out_dir = os.path.join(base_out_dir, fill_mode)
        run_one_experiment(
            fill_mode=fill_mode,
            out_dir=out_dir,
            csv_path=csv_path,
            device=device,
        )


if __name__ == "__main__":
    main()