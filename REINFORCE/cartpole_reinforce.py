import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.prob_history = []
        self.reward_history = []

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

        learning_rate = 0.0002
        self.gamma = 0.98
        self.optim = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
    
    def save_reward_history(self, reward):
        self.reward_history.append(reward)

    def save_prob_history(self, prob):
        self.prob_history.append(prob)
    
    def reset_history(self):
        self.prob_history = []
        self.reward_history = []

    def act(self, state: np.ndarray):
        x = torch.from_numpy(state.astype(np.float32))
        prob = self.forward(x)
        prob_distribution = Categorical(prob)
        action = prob_distribution.sample()
        self.save_prob_history(prob[action])
        return action.item()

    def train_net(self):
        Gt = 0
        self.optim.zero_grad()
        for r, prob in zip(self.reward_history[::-1], self.prob_history[::-1]):
            Gt = r + self.gamma * Gt
            loss = -Gt * torch.log(prob)
            loss.backward()
        self.optim.step()
        self.reset_history()



def main(render=False):
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    print(f"Observation space: {env.observation_space.shape[0]}")
    print(f"Action space: {env.action_space.n}")
    pi = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)

    print_interval = 100
    score = 0.0

    for n_episode in range(5000):
        
        s, _ = env.reset()
        done = False

        while not done:
            action = pi.act(s)
            s_prime, reward, done, info, _ = env.step(action)
            pi.save_reward_history(reward)
            s = s_prime
            score += reward

        pi.train_net()

        if n_episode % print_interval == 0:
            print(f"Episode: {n_episode} \t avg reward: {score/print_interval}")
            score = 0.0
    env.close()
            
if __name__ == "__main__":
    main(True)