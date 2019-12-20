import gym
import matplotlib.pyplot as plt
import torch

N_EPISODES = 10

def run_episode(net, env):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
    episode_reward = 0
    terminal = False
    while not terminal:
        env.render(mode='human')
        output = net(state)
        action_index = int(torch.argmax(output))
        state, reward, terminal, _ = env.step(action_index)
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        episode_reward += reward
    return episode_reward

if __name__ == "__main__":
    # 'policy_net_hardscalingr_lr5e-05_bs512_tu10000_it400000_g0.999_ed0.5_c400000' is not so bad
    # 'policy_net_scalingr_lr0.0001_bs512_tu10000_it400000_g0.999_c300000' best atm
    policy_net = torch.load('saved_models/policy_net_scalingr_lr0.0001_bs512_tu10000_it400000_g0.999_c300000.pt').double()
    policy_net.eval()
    env = gym.make('CartPole-v1')
    episode_x = []
    reward_y = []
    plt.ion()
    fig, ax = plt.subplots()
    iterations = 0
    for episode in range(N_EPISODES):
        episode_reward = run_episode(policy_net, env)
        episode_x.append(episode + 1)
        reward_y.append(episode_reward)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward', color='red')
        ax.plot(episode_x, reward_y, color='red')
        ax.tick_params(axis='y', labelcolor='red')
    env.close()
    plt.ioff()
    plt.show()
