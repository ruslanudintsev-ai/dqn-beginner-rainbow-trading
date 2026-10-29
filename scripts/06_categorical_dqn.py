"""
06_categorical_dqn.py

Categorical DQN (C51) для CartPole-v1.
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_data_dir():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


class CategoricalDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, atoms=51):
        super(CategoricalDQNNetwork, self).__init__()
        self.atoms = atoms
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(128, output_dim * atoms)

    def forward(self, x):
        x = self.fc(x)
        x = self.value_stream(x)
        x = x.view(-1, self.output_dim, self.atoms)
        return torch.softmax(x, dim=2)


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


class CategoricalDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        atoms=51,
        v_min=-10,
        v_max=10,
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        target_update_freq=50,
        buffer_size=50000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atoms - 1)
        self.support = torch.linspace(v_min, v_max, atoms)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = CategoricalDQNNetwork(state_dim, action_dim, atoms).to(self.device)
        self.target_net = CategoricalDQNNetwork(state_dim, action_dim, atoms).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state_t)
            q_values = torch.sum(probs * self.support.to(self.device), dim=2)
        return int(torch.argmax(q_values, dim=1).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        dist = self.policy_net(states_t)
        dist = dist.gather(1, actions_t.unsqueeze(2).expand(self.batch_size, 1, self.atoms)).squeeze(1)

        with torch.no_grad():
            next_dist = self.target_net(next_states_t)
            next_q = torch.sum(next_dist * self.support.to(self.device), dim=2)
            next_actions = torch.argmax(next_q, dim=1)
            next_dist = next_dist[range(self.batch_size), next_actions]

            tz = rewards_t.unsqueeze(1) + self.gamma * (1 - dones_t.unsqueeze(1)) * self.support.to(self.device).unsqueeze(0)
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros(self.batch_size, self.atoms).to(self.device)
            for i in range(self.batch_size):
                for j in range(self.atoms):
                    lower = l[i, j]
                    upper = u[i, j]
                    prob = next_dist[i, j]
                    if lower == upper:
                        m[i, lower] += prob
                    else:
                        m[i, lower] += prob * (upper - b[i, j])
                        m[i, upper] += prob * (b[i, j] - lower)

        loss = -torch.sum(m * torch.log(dist + 1e-8), dim=1).mean()

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

    agent = CategoricalDQNAgent(state_dim, action_dim)

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

    pd.DataFrame({'episode': range(1, len(rewards_history) + 1), 'reward': rewards_history}).to_csv(os.path.join(data_dir, '06_categorical_rewards.csv'), index=False)
    pd.DataFrame(avg_history, columns=['episode', 'avg20']).to_csv(os.path.join(data_dir, '06_categorical_avg20.csv'), index=False)
    stats = {'best_reward': np.max(rewards_history), 'mean_reward': np.mean(rewards_history), 'std_reward': np.std(rewards_history), 'episodes': episodes}
    pd.DataFrame([stats]).to_csv(os.path.join(data_dir, '06_categorical_stats.csv'), index=False)
    torch.save(agent.policy_net.state_dict(), os.path.join(data_dir, '06_categorical_cartpole.pth'))
    print('Training finished for C51. Metrics saved.')


if __name__ == '__main__':
    train(episodes=500)
