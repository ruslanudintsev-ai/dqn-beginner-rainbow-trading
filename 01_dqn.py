"""
01_dqn.py

Простой базовый DQN на CartPole-v1 для обучения архитектуре.
Это адаптированный минимальный пример, чтобы подтвердить структуру перед переходом к Rainbow.
Очень подробные комментарии в коде.
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


def make_data_dir():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    data_dir = os.path.join(base, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# 1) Нейросетевая модель Q-функции
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        # Простая полносвязная сеть
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 2) Replay Buffer
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


# 3) DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        target_update_freq=100,
        buffer_size=100_000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
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


def train(env_name="CartPole-v1", episodes=500):
    data_dir = make_data_dir()
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    rewards_history = []

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
            print(f"Episode {episode}, avg reward: {avg_reward:.2f}, epsilon: {epsilon:.3f}")

    env.close()

    df_rewards = pd.DataFrame({"episode": list(range(1, len(rewards_history) + 1)), "reward": rewards_history})
    df_rewards.to_csv(os.path.join(data_dir, "01_dqn_rewards.csv"), index=False)

    avg_history = [(ep, np.mean(rewards_history[max(0, ep-20):ep])) for ep in range(20, len(rewards_history) + 1, 20)]
    df_avg = pd.DataFrame(avg_history, columns=["episode", "avg20"])
    df_avg.to_csv(os.path.join(data_dir, "01_dqn_avg20.csv"), index=False)

    stats = {
        "best_reward": np.max(rewards_history),
        "mean_reward": np.mean(rewards_history),
        "std_reward": np.std(rewards_history),
        "episodes": len(rewards_history),
    }
    pd.DataFrame([stats]).to_csv(os.path.join(data_dir, "01_dqn_stats.csv"), index=False)

    torch.save(agent.policy_net.state_dict(), os.path.join(data_dir, "01_dqn_cartpole.pth"))
    print("Training finished. Model saved to 01_dqn_cartpole.pth")
    print("Metrics saved: 01_dqn_rewards.csv, 01_dqn_avg20.csv, 01_dqn_stats.csv")


if __name__ == "__main__":
    train(episodes=500)
