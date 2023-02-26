import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        x = F.sigmoid(self.fc2(x))
        return torch.Tensor(x)
    
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
        self.save_prob_history(prob)
        return prob.item()

    def train_net(self):
        self.optim.zero_grad()
        for gt, prob in zip(self.reward_history[::-1], self.prob_history[::-1]):
            loss = -gt * torch.log(prob)
            loss.backward()
        self.optim.step()
        self.reset_history()



def main(render=False):
    if render:
        env = gym.make('MountainCarContinuous-v0', render_mode='human')
    else:
        env = gym.make('MountainCarContinuous-v0')

    pi = PolicyNetwork(env.observation_space.shape[0], 1)

    print_interval = 100
    score = 0.0

    for n_episode in range(5000):
        s, _ = env.reset()
        done = False

        while not done:
            action = pi.act(s)
            action = np.array([action])
            s_prime, reward, terminate, truncated, _ = env.step(action)
            if terminate or truncated:
                done = True
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