"""
Rollout d'evaluation de la politique apprise. 
"""
import numpy as np
import torch

from scripts.train_loop import collect_trajectory


def evaluate_policy(env, actor, device, n_episodes=100, deterministic=True):
    """
    Evaluation multi-épisodes PPO.
    Retourne liste de dict épisode.
    """

    actor.eval()
    results = []

    for ep in range(n_episodes):

        state = env.reset()
        done = False

        inventory = []
        cash = []
        mtm = []
        rewards = []
        spreads = []

        while not done:

            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = actor.sample_action(s)

            action = action.cpu().numpy()[0]

            if deterministic:
                action = np.clip(action, -1, 1)

            next_state, reward, done, info = env.step(action)

            inventory.append(env.inventory)
            cash.append(env.cash)
            mtm.append(env.cash + env.inventory * env.mid)
            rewards.append(reward)
            spreads.append(float(action[0]))

            state = next_state

        results.append({
            "inventory": np.array(inventory),
            "cash": np.array(cash),
            "mtm": np.array(mtm),
            "rewards": np.array(rewards),
            "spreads": np.array(spreads),
            "final_pnl": mtm[-1],
            "cum_reward": np.sum(rewards),
        })

    return results

def run_one_episode(
    env,
    actor,
    device,
    random_reset=True,
    max_steps=200
):
    """
    Episode d'évaluation sans apprentissage.
    """

    actor.eval()

    traj = collect_trajectory(
        env=env,
        model=actor,
        device=device,
        reset_rdm=random_reset,
        max_steps=max_steps
    )

    rewards = traj["rewards"].detach().cpu().numpy()

    inventory = []
    cash = []
    mtm = []

    # reconstruction via infos env
    # on relance proprement un rollout manuel
    if random_reset:
        state = env.reset_random(max_steps)
    else:
        state = env.reset()

    done = False

    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = actor.sample_action(s)

        state, reward, done = env.step(action)

        inventory.append(float(env.inventory))
        cash.append(float(env.cash))
        mtm.append(float(env.cash + env.inventory * env.mid))

    return {
        "inventory": np.array(inventory),
        "cash": np.array(cash),
        "mtm": np.array(mtm),
        "rewards": rewards,
        "cum_reward": rewards.sum(),
        "final_pnl": mtm[-1],
    }