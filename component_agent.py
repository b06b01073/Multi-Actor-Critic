import numpy as np

class ComponentAgent:
    def __init__(self):

        # TODO: build the nn model
        self.net = None

    def take_action(self, state):
        # TODO: select action based on the model output
        return np.random.uniform(low=-1.0, high=1)
    
    def learn(self):
        # TODO: update the model params

        pass