from abc import ABC, abstractmethod
import random
import numpy as np

class Decay(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_decay(self):
        pass

class StepDecay(Decay):
    def __init__(self, initial=1, decay=0.01, minimum=0.02) -> None:
        self.cur_e = initial
        self.decay_amount = decay
        self.minimum = minimum

    def step(self) -> None:
        self.cur_e - self.decay_amount
    
    def get_decay(self) -> float:
        return max(self.cur_e, self.minimum)



class Egreedy():
    def __init__(self, decay_strategy: Decay) -> None:
        self.decay_strategy = decay_strategy

    def step(self):
        self.decay_strategy.step()

    def get_epsilon(self):
        return self.decay_strategy.get_decay()
    
    def sample_action(self, values):
        if random.random() < self.get_epsilon():
            return np.random.randint(len(values)) # size of action space
        else:
            return np.argmax(values)