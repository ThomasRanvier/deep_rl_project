import matplotlib.pyplot as plt
import torch
from atari import Atari
import sys

N_EPISODES = 1
MEAN_K = 10
DISPLAY = False
DYNAMIC = False
SAVE_GIF = True

def run_episode(net, env):
    env.reset()
    episode_reward = 0
    terminal = False
    terminal_life_loss = False
    while not terminal:
        if DISPLAY:
            env.render(mode='human')
        if not terminal_life_loss:
            output = net(env.get_state())
            action_index = int(torch.argmax(output)) + 1
        else:
            action_index = 1
        _, reward, terminal, terminal_life_loss = env.step(action_index)
        episode_reward += reward
    return episode_reward

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = sys.argv[1]
    policy_net = torch.load(model_path).double()
    policy_net.eval()
    env = Atari(device, save_gif=SAVE_GIF)
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
    if DYNAMIC:
        plt.ioff()
    else:
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward', color='red')
        ax.plot(ep, rew, color='orange', alpha=.35)
        ax.plot(ep_mean, rew_mean, color='red')
        ax.tick_params(axis='y', labelcolor='red')
    plt.show()
