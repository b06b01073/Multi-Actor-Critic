import numpy as np

class ComponentAgent:
    def __init__(self, asset):
        self.asset = asset

        # TODO: build the nn model
        self.net = None

    def take_action(self, state):
        # TODO: select action based on the model output
        action = np.random.uniform(low=-1.0, high=1)
        invested_asset = self.asset * np.abs(action)
        self.asset -= invested_asset

        return action, invested_asset
    
    def learn(self):
        # TODO: update the model params

        pass

    def update_asset(self, earning):
        self.asset += earning