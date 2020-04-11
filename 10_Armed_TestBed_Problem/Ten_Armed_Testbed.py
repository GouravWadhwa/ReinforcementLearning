import numpy as np

class TenArmedTestbed (object) :
    def __init__ (self) :
        self.beds_mean = [np.random.randint (-30000, 30000) / 10000.0 for i in range(10)]

    def reward (self, i) :
        return np.random.normal() + self.beds_mean[i]
    
    def all_rewards (self) :
        return [np.random.normal () + self.beds_mean[i] for i in range (10)]

    def best_reward (self) :
        print (self.beds_mean.index (max (self.beds_mean)))
        print (max (self.beds_mean))