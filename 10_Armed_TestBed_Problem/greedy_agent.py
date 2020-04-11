import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from Ten_Armed_Testbed import TenArmedTestbed

def argmax (q_values) :
    top_value = float ('-inf')
    ties = []

    for i in range (len (q_values)) :
        if q_values[i] > top_value :
            ties.clear()
            ties.append (i)
            top_value = q_values[i]
        elif q_values[i] == top_value :
            ties.append (i)

    return np.random.choice (ties)

class GreedyAgent (object) :
    def __init__ (self, arms) :
        self.q_values = np.zeros (arms) + 100
        self.arm_count = np.zeros (arms)
        self.last_action = 0

    def agent_step (self, reward) :
        self.arm_count[self.last_action] += 1
        current_action = argmax (self.q_values)

        self.q_values[self.last_action] += (1.0 / self.arm_count[self.last_action]) * (reward - self.q_values[self.last_action])        
        self.last_action = current_action
        
        return current_action

NUM_RUNS = 500
NUM_STEPS = 1000

agent = GreedyAgent (10)
problem = TenArmedTestbed ()

all_averages = []

for run in tqdm (range (NUM_RUNS)) :
    action = 0
    scores = [0]
    averages = []
    for step in range (NUM_STEPS) :
        current_reward = problem.reward (action)
        action = agent.agent_step (current_reward)

        scores.append(scores[-1] + current_reward)
        averages.append(scores[-1] / (step + 1))
    all_averages.append(averages)

problem.best_reward()

plt.plot (np.mean(all_averages, axis=0))
plt.xlabel ('STEPS')
plt.ylabel ("AVERAGE REWARD")
plt.show()
