import numpy as np

from tile_coding import Tiles

class Agent () :
    def agent_init (self, agent_info={}) :
        self.num_tiles = agent_info.get("num_tiles", 8)
        self.num_boxes = agent_info.get("num_boxes", 8)
        self.num_actions = agent_info.get("num_actions", 3)
        
        self.epsilon = agent_info.get("epsilon", 0.0)
        self.gamma = agent_info.get("gamma", 1.0)
        self.alpha = agent_info.get("alpha", 0.5) / self.num_tiles

        self.max_value_x = agent_info.get ("max_value_x", 0.7)
        self.min_value_x = agent_info.get ("min_value_x", -1.2)
        self.max_value_y = agent_info.get ("max_value_y", 0.07)
        self.min_value_y = agent_info.get ("min_value_y", -0.07)
        
        self.initial_weights = agent_info.get("initial_weights", 0.0)
        self.w = np.ones((self.num_actions, self.num_tiles * self.num_boxes * self.num_boxes)) * self.initial_weights

        self.Tiles = Tiles (
            self.min_value_x,
            self.max_value_x,
            self.min_value_y,
            self.max_value_y,
            self.num_tiles,
            self.num_boxes,
            self.num_boxes
        )

    def argmax (self, q_values) :
        best = float ("-inf")
        ties = []

        for i in range (len (q_values)) :
            if q_values[i] > best :
                best = q_values[i]
                ties = [i]
            elif q_values[i] == best :
                ties.append (i)

        return np.random.choice(ties)

    def agent_select_action (self, tiles) :
        action_values = []
        choosen_action = None

        for i in range (self.num_actions) :
            value = 0
            for j in tiles :
                value += self.w[i][j]
            action_values.append (value)

        if np.random.rand () < self.epsilon :
            chosen_action = np.random.randint (0, self.num_actions)
        else :
            chosen_action = self.argmax (action_values)

        return chosen_action

    def agent_start (self, state) :
        position, velocity = state

        active_tiles = self.Tiles.coded_representation (position, velocity)
        current_action = self.agent_select_action (active_tiles)

        self.last_action = current_action
        self.previous_tiles = active_tiles

        return self.last_action

    def agent_step (self, reward, state) :
        position, velocity = state

        active_tiles = self.Tiles.coded_representation (position, velocity)
        current_action = self.agent_select_action (active_tiles)
        
        q_earlier = np.sum ([self.w[self.last_action][j] for j in self.previous_tiles])
        q_current = np.sum ([self.w[current_action][j] for j in active_tiles])
        
        grad_q = np.zeros (self.w.shape)
        
        for t in self.previous_tiles :
            grad_q[self.last_action][t] = 1 
        
        self.w = self.w + self.alpha * (reward + self.gamma * q_current - q_earlier) * grad_q
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        
        return self.last_action
    
    def agent_end (self, reward) :
        q_earlier = np.sum ([self.w[self.last_action][j] for j in self.previous_tiles])
        
        grad_q = np.zeros (self.w.shape)
        for t in self.previous_tiles :
            grad_q[self.last_action][t] = 1 
        
        self.w = self.w + self.alpha * (reward - q_earlier) * grad_q