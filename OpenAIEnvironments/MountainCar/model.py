import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from environment import Environment
from agent import Agent
from tile_coding import Tiles

agent_info = {
    "num_tiles":8,
    "num_boxes":8,
    "num_actions":3, 
    "min_value_x":-1.2,
    "max_value_x":0.5,
    "min_value_y":-0.07,
    "max_value_y":0.07,
    "epsilon":0.1,
    "gamma":1.0,
    "alpha":0.5
}

RUNS = 5
EPISODES = 500

agent = Agent()
environment = Environment()

average_rewards_sum = []
average_state_values = []
agent_rmsve = np.zeros ((RUNS, EPISODES))

count = 0
for run in range (RUNS) :
    agent.agent_init (agent_info)
    environment.env_init ()
    
    count += 1
    rewards_sum = []

    for episode in tqdm (range (EPISODES)) :
        last_state = environment.env_start ()
        last_action = agent.agent_start (last_state)
        
        total_reward = 0
        is_terminal = False

        while not is_terminal :
            last_state, reward, is_terminal = environment.env_step (last_action)
            total_reward += reward
            
            if is_terminal :
                agent.agent_end (reward)
            else :
                last_action = agent.agent_step (state=last_state, reward=reward)

            # print (last_action)
            # print (last_state)

        rewards_sum.append (total_reward * -1)
    
    last_state = environment.env_start ()
    last_action = agent.agent_start (last_state)

    environment.render ()

    total_reward = 0
    is_terminal = False

    while not is_terminal :
        last_state, reward, is_terminal = environment.env_step (last_action)
        total_reward += reward

        environment.render ()

        if is_terminal :
            agent.agent_end (reward)
        else :
            last_action = agent.agent_step (state=last_state, reward=reward)

    average_rewards_sum.append (rewards_sum)

plt.plot (np.mean (average_rewards_sum, axis=0))
plt.show()
plt.clf()