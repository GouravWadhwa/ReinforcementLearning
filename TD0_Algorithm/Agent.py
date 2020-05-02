import numpy as np

class agent () :
    def agent_init (self, agent_info={}) :
        self.policy = agent_info.get ("policy")
        self.discount = agent_info.get ("discount")
        self.step_size = agent_info.get ("step_size")

        self.values = np.zeros ((self.policy.shape[0],))

    def agent_start (self, state) :
        action = np.random.choice (range (self.policy.shape[1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_step (self, reward, state) :
        target = reward + self.values [state] * self.discount
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state]) 

        action = np.random.choice (range (self.policy.shape[1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_end  (self, reward) :
        target = reward
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state]) 

    def agent_values (self) :
        return self.values

    def agent_cleanup(self):
        self.last_state = None