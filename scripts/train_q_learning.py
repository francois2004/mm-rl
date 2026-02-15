from src.env_toy_mm import MMSimulator
from src.utils.discretisation import state_index
from src.agents.q_learning import Qlearning_agent
import numpy as np

env = MMSimulator("data/raw/toy_lob.csv", p_fill_base = .30, eta_inv = 1e-6)
n_states = 65
actions = [.0005, .001, .0015, .002, .003, .004, .005]
n_action = len(actions)
agent = Qlearning_agent(
    n_states,
    actions, 
    alpha = .1,
    gamma = .99,
    epsilon = 1.
)
nb_episode = 200
T_max = 200
avg_reward = 0
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
        agent.update(s_idx, a_idx, reward, s_next_idx, done)
        visited.add(s_idx)
        s_idx = s_next_idx
        total_reward += reward
        i+=1
        avg_reward+=reward
    agent.epsilon = np.maximum(agent.epsilon * .995, .05)
    if episode%50 ==0:
    #     #last_state = env._state()
    #     #mid_last = last_state[0]
    #     #mtm_final = env.cash + env.inventory * mid_last
    #     #total_trades = env.nb_trades
    #     #print ("episode: ",episode, "total reward: " , total_reward, "inventory: ", env.inventory, "cash: ", env.cash)
    #     #print("last mid: ", mid_last, "mtm final: ", mtm_final, "total_trades: ", total_trades)
         mean_abs_Q = np.mean(np.abs(agent.Q))
         n_visited_states = len(visited)
         print("episode: ", episode)
         print("epsilon: ", agent.epsilon, "mean abs Q: ", mean_abs_Q, "n_visited_states: ", n_visited_states, "average reward last ten occurences: ", avg_reward/10)
    #     avg_reward = 0


np.save("models/q_table.npy", agent.Q)











