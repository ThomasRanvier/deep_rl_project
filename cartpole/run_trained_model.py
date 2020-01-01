import gym
import matplotlib.pyplot as plt
import torch

N_EPISODES = 1000
MEAN_K = 10
DISPLAY = False
DYNAMIC = False
MODEL_PATH = 'best_models/perfect_model/model.pt' # Always 500
#MODEL_PATH = 'best_models/good_model/model.pt' # Usually around 450
#MODEL_PATH = 'best_models/ok_model/model.pt' # Usually around 300

def run_episode(net, env):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
    episode_reward = 0
    terminal = False
    while not terminal:
        if DISPLAY:
            env.render(mode='human')
        output = net(state)
        action_index = int(torch.argmax(output))
        state, reward, terminal, _ = env.step(action_index)
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        episode_reward += reward
    return episode_reward

if __name__ == "__main__":
    policy_net = torch.load(MODEL_PATH).double()
    policy_net.eval()
    env = gym.make('CartPole-v1')
    ep = []
    rew = []
    ep_mean = []
    rew_mean = []
    iterations = 0
    reward_sum = 0
    total = 0
    if DYNAMIC:
        plt.ion()
    fig, ax = plt.subplots()
    for episode in range(N_EPISODES):
        episode_reward = run_episode(policy_net, env)
        reward_sum += episode_reward
        total += episode_reward
        rew.append(episode_reward)
        ep.append(episode + 1)
        if episode % MEAN_K == MEAN_K - 1:
            print('Episode', episode + 1)
            ep_mean.append(episode + 1)
            rew_mean.append(reward_sum / MEAN_K)
            reward_sum = 0
            if DYNAMIC:
                ax.set_xlabel('Episodes')
                ax.set_ylabel('Reward', color='red')
                ax.plot(ep, rew, color='orange', alpha=.35)
                ax.plot(ep_mean, rew_mean, color='red')
                ax.tick_params(axis='y', labelcolor='red')
    print('Reward mean:', total / N_EPISODES)
    env.close()
    if DYNAMIC:
        plt.ioff()
    else:
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward', color='red')
        ax.plot(ep, rew, color='orange', alpha=.35)
        ax.plot(ep_mean, rew_mean, color='red')
        ax.tick_params(axis='y', labelcolor='red')
    plt.show()
