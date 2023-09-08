import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from core import train, play

train_env = gym.make('CliffWalking-v0')
play_env = gym.make('CliffWalking-v0', render_mode='human')

class RLAgent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = 0.9
        self.gamma = 0.9
        self.epsilon = 0.1
        self.sample_count = 0

    def choose_action(self, ob):
        # 贪心算法
        # 递减探索概率，越往后面探索的欲望越低
        self.sample_count += 1
        epsilon = 1.0 / self.sample_count
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.n_actions)

        return np.argmax(self.q_table[ob])

    def update(self, ob, action, reward, next_ob, done):
        predict = self.q_table[ob][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_ob])
        self.q_table[ob][action] += self.lr * (target - predict)
    
    def get_parameters(self):
        return self.q_table

agent = RLAgent(train_env.observation_space.n, train_env.action_space.n)

train(train_env, agent)
play(play_env, agent)
