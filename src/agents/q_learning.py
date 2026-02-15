import numpy as np


Delta = [.0005, .001, .0015, .002, .003, .004, .005]
n_action = len(Delta)

class Qlearning_agent : 
    def __init__(self, n_states, actions, alpha, gamma, epsilon):
        self.Q = np.zeros((n_states, len(actions)))
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_actions(self, s_idx): 
        u = np.random.uniform(0,1)

        if u < self.epsilon:
            a_idx = np.random.randint(0, len(self.actions))
        else : 
            a_idx = np.argmax(self.Q[s_idx,:])

        delta = self.actions[a_idx]
        return int(a_idx), delta
    
    def update(self,s_idx, a_idx, reward, s_next_idx, done):
        current = self.Q[s_idx, a_idx]
        if done : 
            target = reward
        else: 
            target = reward + self.gamma * np.max(self.Q[s_next_idx, :])
        self.Q[s_idx, a_idx] = current + self.alpha *(target - current)


