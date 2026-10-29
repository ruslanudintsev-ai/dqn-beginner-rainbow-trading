"""
04_dueling.py

Dueling DQN вариант: отдельные потоки для V(x) и A(x,a).
"""

import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------- reproducibility -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------


def make_data_dir():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=50,
        buffer_size=50_000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

        self.steps_done = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q_values = rewards_t + self.gamma * (1 - dones_t) * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def train(env_name='CartPole-v1', episodes=500):
    data_dir = make_data_dir()
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    rewards_history = []
    avg_history = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0.0
        done = False

        while not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1.0 * episode / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)

        if episode % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_history.append((episode, avg_reward))
            print(f'Episode {episode}, avg reward: {avg_reward:.2f}, epsilon: {epsilon:.3f}')

    env.close()

    pd.DataFrame({'episode': list(range(1, len(rewards_history) + 1)), 'reward': rewards_history}).to_csv(
        os.path.join(data_dir, '04_dueling_rewards.csv'), index=False
    )
    pd.DataFrame(avg_history, columns=['episode', 'avg20']).to_csv(
        os.path.join(data_dir, '04_dueling_avg20.csv'), index=False
    )

    stats = {
        'best_reward': np.max(rewards_history),
        'mean_reward': np.mean(rewards_history),
        'std_reward': np.std(rewards_history),
        'episodes': episodes,
    }
    pd.DataFrame([stats]).to_csv(os.path.join(data_dir, '04_dueling_stats.csv'), index=False)

    torch.save(agent.policy_net.state_dict(), os.path.join(data_dir, '04_dueling_cartpole.pth'))
    print('Training finished. Model saved to 04_dueling_cartpole.pth')
    print('Metrics saved: 04_dueling_rewards.csv, 04_dueling_avg20.csv, 04_dueling_stats.csv')


if __name__ == '__main__':
    train(episodes=500)
