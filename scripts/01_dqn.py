"""
01_dqn.py — Basic DQN on CartPole-v1
Adapted for the dqn-beginner project (seed, data saving, comparison-friendly).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
import pandas as pd

SEED = 42
EPISODES = 500

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def soft_update_target(self, tau=0.005):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def make_data_dir():
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)

def train(episodes=EPISODES):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    make_data_dir()
    rewards = []
    avg20 = []

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED + ep)
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            agent.soft_update_target()
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
        avg20.append(np.mean(rewards[-20:]))
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.1f} | Avg20: {avg20[-1]:.1f} | ε: {agent.epsilon:.3f}")

    # Save stats
    pd.DataFrame({"episode": range(1, episodes+1), "reward": rewards}).to_csv("../data/01_dqn_rewards.csv", index=False)
    pd.DataFrame({"episode": range(1, episodes+1), "avg20": avg20}).to_csv("../data/01_dqn_avg20.csv", index=False)
    summary = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "best_reward": float(np.max(rewards)),
    }
    pd.DataFrame([summary]).to_csv("../data/01_dqn_stats.csv", index=False)
    print("Saved 01_dqn stats to data/")
    return summary

if __name__ == "__main__":
    train()
