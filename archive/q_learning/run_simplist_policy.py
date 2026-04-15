from src.env_toy_mm import MMSimulator
import numpy as np

sim = MMSimulator("data/raw/toy_lob.csv", p_fill_base = .30, eta_inv = .0001)
state = sim.reset()

total_reward = 0.
done = False

while not done : 
    delta = np.random.uniform(.01, .2)
    state, r, done = sim.step(delta, k = 100)
    total_reward += r
mid_last = state[0]
mtm_final = sim.cash + sim.inventory*mid_last

print("Total reward:", total_reward)
print("Final inventory:", sim.inventory)
print("Final cash:", sim.cash)
print("Mid last:", mid_last)
print("MTM final:", mtm_final)