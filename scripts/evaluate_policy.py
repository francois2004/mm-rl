import numpy as np
from src.envs.env_toy_mm import MMSimulator
from src.utils.discretisation import state_index


def evaluate_policy(agent, n_episodes=500, T_max=200, greedy=True):
    env = MMSimulator("data/raw/toy_lob.csv",
                      p_fill_base=0.30,
                      eta_inv=0.0001)

    rewards = []
    trades = []
    inv_rms = []
    mtm_finals = []
    penalties = []

    for _ in range(n_episodes):

        state = env.reset_random(T_max)
        s_idx = state_index(state)

        total_reward = 0
        done = False
        t = 0

        while (not done) and (t < T_max):

            if greedy:
                a_idx = np.argmax(agent.Q[s_idx])
                delta = agent.actions[a_idx]
            else:
                delta = agent.actions[len(agent.actions)//2]  # baseline: delta constant

            next_state, reward, done = env.step(delta)
            s_idx = state_index(next_state)

            total_reward += reward
            t += 1

        rewards.append(total_reward)
        trades.append(env.nb_trades)
        penalties.append(env.penalty_sum)

        inv_path = np.array(env.inventory_path)
        inv_rms.append(np.sqrt(np.mean(inv_path**2)))

        mtm_final = env.cash + env.inventory * env.data.iloc[env.t]["mid"]
        mtm_finals.append(mtm_final)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_trades": np.mean(trades),
        "mean_inv_rms": np.mean(inv_rms),
        "mean_mtm_final": np.mean(mtm_finals),
        "mean_penalty": np.mean(penalties)
    }

