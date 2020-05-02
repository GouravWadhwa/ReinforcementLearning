import numpy as np

from Environment import environment
from Agent import agent

from tqdm import tqdm

def run_experiments (env_info, agent_info, num_episodes=500, experiment_name=None) :
    env = environment ()
    age = agent ()

    env.env_init(env_info)
    age.agent_init (agent_info)

    for i in range (num_episodes) :
        terminal = False

        last_state = env.env_start ()
        last_action = age.agent_start (last_state)
        total_reward = 0

        while not terminal :
            (reward, last_state, terminal) = env.env_step (last_action)

            total_reward += reward
            if terminal :
                age.agent_end (reward)
            else :
                last_action = age.agent_step (reward, last_state)
        

    values = age.agent_values ()
    print ("VALUE FUNCTION", end="\n\n")

    if experiment_name is not None:
        print (experiment_name)

    for i in range (env_info.get("height", 4)) :
        for j in range (env_info.get ("width", 12)) :
            print ("%7.2f"%values[i*env_info.get("width", 12) + j], end=' ')
        print ()   

env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}

policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [0.9, 0.1/3., 0.1/3., 0.1/3.]
for i in range(24, 35):
    policy[i] = [0.1/3., 0.1/3., 0.1/3., 0.9]
policy[35] = [0.1/3., 0.1/3., 0.9, 0.1/3.]

agent_info.update({"policy": policy})

run_experiments (env_info, agent_info, 1000, "With Close to optimal Policy")


