import numpy as np

class Random():
    def __init__(self) -> None:
        pass
    
    def sample_action(self, value):
        return np.random.randint(len(value))