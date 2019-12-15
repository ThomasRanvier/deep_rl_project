import gym
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch

from replay_memory import ReplayMemory
from net import Net
from agent import Agent
from config import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rm = ReplayMemory(RM_CAPACITY)
    policy_net = Net().double().to(device)
    target_net = Net().double().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    epsilon = INITIAL_EPSILON
    env = gym.make('Breakout-ramNoFrameskip-v4')
    agent = Agent(rm, policy_net, target_net, optimizer, criterion, env, device)
    reward_hist = []
    epsilon_hist = []
    if DYNAMIC_PLOT:
        plt.ion()
    for episode in range(N_EPISODES):
        total_reward = agent.run_episode(epsilon,
                                         update_target_net=(episode % TARGET_UPDATE == 0),
                                         save_nets=(SAVE_MODELS and episode == N_EPISODES - 1))
        reward_hist.append(total_reward)
        epsilon_hist.append(epsilon)
        if VERBOSE:
            print('episode:', episode + 1, '- total reward:', total_reward)
        if DYNAMIC_PLOT:
            plt.figure(1)
            plt.clf()
            plt.title('Training... eps: ' + str(round(epsilon, 2)))
            plt.xlabel('Episode')
            plt.ylabel('Total reward')
            plt.plot(reward_hist)
            plt.plot(epsilon_hist)
        epsilon -= (epsilon > MINIMAL_EPSILON) * EPSILON_ANNEALING_STEP
    env.close()
    if DYNAMIC_PLOT:
        plt.title('Done')
        plt.ioff()
        plt.show()
    else:
        plt.title('Reward evolution')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(reward_hist)
        plt.plot(epsilon_hist)
        plt.savefig(PLOTS_DIR + FILE_SUFFIX + '.png')
