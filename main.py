import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import gymnasium as gym
import random


class ExperienceReplay:
    def __init__(self, maxCapacity):
        self.maxCapacity = maxCapacity
        self.memory = deque(maxlen=self.maxCapacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.gamma = 0.95  # Adjusted gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # Adjusted epsilon decay
        self.learning_rate = 0.0005  # Adjusted learning rate
        self.memory = ExperienceReplay(10000)  # Reduced memory capacity
        self.model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_model_update_freq = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        state_array = np.array(state[0]) if isinstance(state, tuple) else state
        return np.argmax(self.model(torch.tensor(state_array).float().unsqueeze(0)).detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        #state = [np.array(s) for s in state]
        state = torch.tensor(np.array(state)).float()
        action = torch.tensor(np.array(action)).long()
        reward = torch.tensor(np.array(reward)).float()
        next_state = torch.tensor(np.array(next_state)).float()
        done = torch.tensor(np.array(done)).float()
        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1).values
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, batch_size):
        for episode in range(episodes):
            state = self.env.reset()
            terminated = False
            truncated = False
            done = False
            counter = 0
            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if counter == 2:
                    pass

                state = next_state[0] if isinstance(next_state, tuple) else next_state

                self.memory.push(state, action, reward, next_state, done)

                if not isinstance(next_state, np.ndarray):
                    pass
                state = next_state
                self.replay(batch_size)

                counter += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if episode % self.target_model_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            print(f'Episode: {episode + 1}/{episodes}, Epsilon: {self.epsilon}')

    def play(self, test_times=1):
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.model.eval()
        for i in range(test_times):
            state = self.env.reset()
            terminated = False
            truncated = False
            done = False
            total_reward = 0
            while not done:
                self.env.render()
                state = state[0] if isinstance(state, tuple) else state
                action = np.argmax(self.model(torch.tensor(state).float().unsqueeze(0)).detach().numpy())
                state, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += rew
            print(f'Test {i + 1}/{test_times}, Total Reward: {total_reward}')
        self.env.close()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()


if __name__ == '__main__':
    agent = Agent()
    agent.train(1000, 64)
    agent.play(5)
    agent.save('model1.pth')
    while True:
        agent.play(int(input('Enter the number of times to play: \n')))
