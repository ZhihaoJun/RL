import matplotlib.pyplot as plt

def train(env, agent, train_eps=500):
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
    
    print(agent.get_parameters())

    # 画图
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.show()

def play(env, agent):
    ob, info = env.reset()
    while True:
        action = agent.choose_action(ob)
        next_ob, reward, done, truncated, info = env.step(action)
        if done:
            break
        ob = next_ob
        env.render()
