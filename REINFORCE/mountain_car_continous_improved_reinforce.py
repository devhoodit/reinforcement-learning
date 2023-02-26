import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.0002, gamma=0.98) -> None:
        super().__init__()

        self.prob_history = []
        self.reward_history = []
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self.gamma = gamma
        self.optim = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
        output = self.forward(x)
        pd = Normal(loc=output[0], scale=torch.sigmoid(output[1]))
        action = pd.sample()
        self.save_prob_history(pd.log_prob(action))
        return action

    def train_net(self):
        self.optim.zero_grad()
        Gt_history = []
        Gt = 0
        for r in self.reward_history[::-1]:
            Gt = r + self.gamma * Gt
            Gt_history.append(Gt)

        b = sum(Gt_history) / len(Gt_history)
        for gt, prob in zip(Gt_history, self.prob_history[::-1]):
            loss = -(gt - b) * prob
            loss.backward()
        self.optim.step()
        self.reset_history()



def main(render=False):
    if render:
        env = gym.make('MountainCarContinuous-v0', render_mode='human')
    else:
        env = gym.make('MountainCarContinuous-v0')

    pi = PolicyNetwork(env.observation_space.shape[0], 2)

    print_interval = 20
    score = 0.0

    for n_episode in range(1500):
        s, _ = env.reset()
        done = False

        while not done:
            action = pi.act(s)
            action = np.array([action])
            s_prime, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated
            pi.save_reward_history(reward)
            s = s_prime
            score += reward

        pi.train_net()
        
        if n_episode % print_interval == 0:
            print(f"Episode: {n_episode} \t avg reward: {score/print_interval}")
            score = 0.0

    env.close()

if __name__ == "__main__":
    main()