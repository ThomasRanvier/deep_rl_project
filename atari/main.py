import gym
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import time

from replay_memory import ReplayMemory
from net import Net
from agent import Agent
from config import *

def display_plot(epsilon_x, epsilon_y, reward_x, reward_y, save = False, ax1 = None, ax2 = None, fig = None):
    if ax1 == None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('epsilon', color='red')
    ax1.plot(epsilon_x, epsilon_y, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('reward per episode', color='blue')
    ax2.plot(reward_x, reward_y, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    fig.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR + FILE_SUFFIX + '.png')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rm = ReplayMemory(RM_CAPACITY)
    policy_net = Net().double().to(device)
    target_net = Net().double().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    env = gym.make('Breakout-ramNoFrameskip-v4')
    agent = Agent(rm, policy_net, target_net, optimizer, criterion, env, device)
    reward_x = [0]
    reward_y = [0]
    epsilon_x = [0, N_ITERATIONS // FACTOR_TO_MIN_EPS, N_ITERATIONS]
    epsilon_y = [INITIAL_EPSILON, MINIMAL_EPSILON, MINIMAL_EPSILON]
    if DYNAMIC_PLOT:
        plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    """
    TO IMPROVE:
        - Switch from Adam optim to MSProp to see difference, find hyperparameters to use
        - Add recording
    """
    iterations = 0
    episode = 1
    init_start = time.time()
    while iterations <= N_ITERATIONS:
        start = time.time()
        epsilon, iterations, episode_reward = agent.run_episode()
        end = time.time()
        reward_x.append(iterations)
        reward_y.append(episode_reward)
        if VERBOSE:
            total_estimated_time = ((end - init_start) / iterations) * N_ITERATIONS
            remaining_estimation = (total_estimated_time - (end - init_start)) / 60
            print('Ep {5} - ite {0}/{1} - reward {2} - eps {3:.2f} - time {4:.2f}s - total {6:.2f}m - remain {7:.2f}m'
                  .format(iterations, N_ITERATIONS, episode_reward, epsilon, end - start, episode,
                          (end - init_start) / 60, remaining_estimation), flush=True)
        if DYNAMIC_PLOT:
            display_plot(epsilon_x, epsilon_y, reward_x, reward_y, ax1=ax1, ax2=ax2, fig=fig)
        episode += 1
    env.close()
    if DYNAMIC_PLOT:
        plt.ioff()
        plt.show()
    display_plot(epsilon_x, epsilon_y, reward_x, reward_y, save=True)
