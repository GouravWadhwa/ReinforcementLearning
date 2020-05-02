import numpy as np

class environment () :
    def env_init (self, env_info={}) :
        reward = None
        state = None
        termination = None

        self.reward_state = (reward, state, termination)

        self.grid_h = env_info.get ("grid_height", 4)
        self.grid_w = env_info.get ("grid_width", 12)

        self.start_location = (self.grid_h - 1, 0)
        self.goal_location = (self.grid_h - 1, self.grid_w - 1)

        self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]

    def state (self, location) :
        return self.grid_w * location[0] + location[1]

    def env_start (self) :
        self.agent_location = self.start_location

        reward = 0
        state = self.state (self.agent_location)
        termination = False

        self.reward_state = (reward, state, termination)

        return self.reward_state[1]

    def env_step (self, action) :
        if action == 0 :                                                                        # UP
            if self.agent_location[0] - 1 >= 0 :
                self.agent_location = (self.agent_location[0] - 1, self.agent_location[1])
        elif action == 1 :                                                                      # LEFT
            if self.agent_location[1] -1 >= 0 :
                self.agent_location = (self.agent_location[0], self.agent_location[1] - 1)
        elif action == 2 :                                                                      # DOWN
            if self.agent_location[0] + 1 < self.grid_h :
                self.agent_location = (self.agent_location[0] + 1, self.agent_location[1])
        elif action == 3 :                                                                      # RIGHT
            if self.agent_location[1] + 1 < self.grid_w :
                self.agent_location = (self.agent_location[0], self.agent_location[1] + 1)
        else :
            raise Exception (str(action) + " is not recognized")

        reward = -1
        termination = False

        if self.agent_location == self.goal_location :
            termination = True
        elif self.agent_location in self.cliff :
            reward = -100
            self.agent_location = self.start_location

        self.reward_state = (reward, self.state(self.agent_location), termination)

        return self.reward_state

    def env_cleanup (self) :
        self.agent_location = self.start_location