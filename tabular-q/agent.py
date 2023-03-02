import numpy as np
from ..explorePolicy import egreedy

class Agent():
    def __init__(self, state_size, action_space) -> None:
        self.state_size = state_size
        self.action_space = action_space
        self.qtable = np.zeros(*self.state_size, len(action_space))
        self.explore_policy = egreedy.Egreedy(decay_strategy=egreedy.StepDecay(initial=1, decay=0.01, minimum=0.01))

    def sample_action(self):
        
        return np.random.randint(self.action_space)
    