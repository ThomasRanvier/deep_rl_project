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

def display_plot(iterations_x, loss_y, epsilon_y, reward_x, reward_y, save = False, ax1 = None, ax2 = None, ax3 = None, fig = None):
    if ax1 == None:
        fig, (ax1, ax3) = plt.subplots(2)
        ax2 = ax1.twinx()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss', color='red')
    ax1.plot(iterations_x, loss_y, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('reward per episode', color='blue')
    ax2.plot(reward_x, reward_y, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax3.set_ylabel('epsilon', color='orange')
    ax3.plot(iterations_x, epsilon_y, color='orange')
    fig.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR + FILE_SUFFIX + '.png')

if __name__ == "__main__":
    #torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Used device: {} - epsilon annealing step: {}'.format(device, EPSILON_ANNEALING_STEP), flush=True)
    rm = ReplayMemory(RM_CAPACITY)
    policy_net_a = Net(FEATURES_SIZES).double().to(device)
    target_net_a = Net(FEATURES_SIZES).double().to(device)
    target_net_a.load_state_dict(policy_net_a.state_dict())
    target_net_a.eval()
    policy_net_b = Net(FEATURES_SIZES).double().to(device)
    target_net_b = Net(FEATURES_SIZES).double().to(device)
    target_net_b.load_state_dict(policy_net_a.state_dict())
    target_net_b.eval()
    optimizer_a = optim.Adam(policy_net_a.parameters(), lr=LEARNING_RATE)
    optimizer_b = optim.Adam(policy_net_b.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()# nn.MSELoss()
    env = gym.make('CartPole-v1')
    agent = Agent(rm, policy_net_a, target_net_a, optimizer_a, policy_net_b, target_net_b, optimizer_b, criterion, env, device)
    reward_x = [0]
    reward_y = [0]
    iterations_x = []
    if DYNAMIC_PLOT:
        plt.ion()
    fig, (ax1, ax3) = plt.subplots(2)
    fig.set_size_inches(10, 8)
    ax2 = ax1.twinx()
    iterations = 0
    episode = 1
    init_start = time.time()
    while iterations < N_ITERATIONS:
        epsilon_y, loss_y, iterations, episode_reward = agent.run_episode()
        end = time.time()
        reward_x.append(iterations)
        reward_y.append(episode_reward)
        iterations_x.extend(list(range(len(iterations_x) + 1, iterations + 1)))
        if VERBOSE:
            total_estimated_time = ((end - init_start) / iterations) * N_ITERATIONS
            remaining_estimation = (total_estimated_time - (end - init_start)) / 60
            print(
                'Ep {5} - ite {0}/{1} - reward {2} - eps {3:.2f} - loss {8:.2f} - rm load {4:.2f}% - uptime {6:.2f}m - remaining {7:.2f}m'
                .format(iterations, N_ITERATIONS, int(round(episode_reward)), epsilon_y[-1], 100. * len(rm) / RM_CAPACITY,
                        episode, (end - init_start) / 60, remaining_estimation, loss_y[-1]), flush=True)
        if DYNAMIC_PLOT:
            display_plot(iterations_x, loss_y, epsilon_y, reward_x, reward_y, ax1=ax1, ax2=ax2, ax3=ax3, fig=fig)
        episode += 1
    env.close()
    if DYNAMIC_PLOT:
        plt.ioff()
        plt.show()
    display_plot(iterations_x, loss_y, epsilon_y, reward_x, reward_y, save=True)
