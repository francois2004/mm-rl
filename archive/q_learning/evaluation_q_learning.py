from src.agents.q_learning import Qlearning_agent
from src.env_toy_mm import MMSimulator
from src.utils.discretisation import state_index
import numpy as np

Q = np.load("models/q_table.npy")
agent = Qlearning_agent(
    n_states = 55,
    actions = [.0005, .001, .0015, .002, .003, .004, .005],
    alpha = .1,
    gamma = .99,
    epsilon = 0,
)
env = MMSimulator("data/raw/toy_lob.csv", p_fill_base = .30, eta_inv = 1e-6)
agent.Q = Q
T_max = 200
nb_episode = 500
delta = .002
T_R_baseline = []
for episode in range (nb_episode): 

    state = env.reset_random(T_max)
    s_idx = state_index(state)
    total_reward = 0
    done = False

    i = 0
    visited = set()
    while (not done) and (i<T_max) : 
        #a_idx, delta = agent.select_actions(s_idx)

        next_state, reward, done = env.step(delta)
        s_next_idx = state_index(next_state)
        #agent.update(s_idx, a_idx, reward, s_next_idx, done)
        visited.add(s_idx)
        s_idx = s_next_idx
        total_reward += reward
        i+=1

    T_R_baseline.append(total_reward)
T_R_baseline = np.asarray(T_R_baseline)
print("BASELINE")
print("mean reward: ", np.mean(T_R_baseline), "standard deviation", np.std(T_R_baseline))

T_Rew = []
for episode in range (nb_episode): 

    state = env.reset_random(T_max)
    s_idx = state_index(state)
    total_reward = 0
    done = False

    i = 0
    visited = set()
    while (not done) and (i<T_max) : 
        a_idx, delta = agent.select_actions(s_idx)

        next_state, reward, done = env.step(delta)
        s_next_idx = state_index(next_state)
        #agent.update(s_idx, a_idx, reward, s_next_idx, done)
        visited.add(s_idx)
        s_idx = s_next_idx
        total_reward += reward
        i+=1

    T_Rew.append(total_reward)
T_Rew = np.asarray(T_Rew)
print("Q LEARNED")
print("mean reward: ", np.mean(T_Rew), "standard deviation", np.std(T_Rew))


from scripts.diagnostics import evaluate_policy

print("BASELINE")
baseline_stats = evaluate_policy(agent, greedy=False)
for k, v in baseline_stats.items():
    print(k, ":", v)

print("\nQ LEARNED")
learned_stats = evaluate_policy(agent, greedy=True)
for k, v in learned_stats.items():
    print(k, ":", v)