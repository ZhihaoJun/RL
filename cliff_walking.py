import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')

rewards = []

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

agent = RLAgent(env.observation_space.n, env.action_space.n)

def train():
    train_eps = 500
    rewards = []
    ma_rewards = []

    for ep in range(train_eps):
        print(f'ep {ep}')
        ep_reward = 0
        ob, info = env.reset()
        
        while True:
            action = agent.choose_action(ob)
            next_ob, reward, done, truncated, info = env.step(action)
            agent.update(ob, action, reward, next_ob, done)

            ob = next_ob
            ep_reward += reward
            if done:
                break
        print(f'ep {ep} ends with reward:{ep_reward}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
    
    print(agent.q_table)

    # 画图
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.show()

def play():
    play_env = gym.make('CliffWalking-v0', render_mode='human')
    ob, info = play_env.reset()
    while True:
        action = agent.choose_action(ob)
        next_ob, reward, done, truncated, info = play_env.step(action)
        if done:
            break
        ob = next_ob
        env.render()

train()
play()
