import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import time

from atari import Atari
from replay_memory import ReplayMemory
from net import Net
from agent import Agent
from config import *

def display_plot(iterations_x, loss_y, epsilon_y, episode_x, reward_y, save = False, ax1 = None, ax2 = None, ax3 = None, fig = None):
    if ax1 == None:
        fig, (ax1, ax3) = plt.subplots(2)
        ax2 = ax1.twinx()
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss', color='red')
    ax1.plot(episode_x, loss_y, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('reward per episode', color='blue')
    ax2.plot(episode_x, reward_y, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax3.set_ylabel('epsilon', color='orange')
    ax3.plot(iterations_x, epsilon_y, color='orange')
    fig.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR + FILE_SUFFIX + '.png')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Used device: {}'# - eps annealing step 1: {} - eps annealing step 2: {}'
          .format(device), flush=True)#, EPSILON_ANNEALING_STEP_1, EPSILON_ANNEALING_STEP_2), flush=True)
    print('Total iterations: {} - rm capacity: {} - bs: {}'
          .format(N_ITERATIONS, RM_CAPACITY, MINIBATCH_SIZE), flush=True)
    rm = ReplayMemory(device)
    #policy_net = Net(heavy_model=True).double().to(device)
    policy_net = torch.load('best_models/rewclip_lr0000625/model.pt').double().to(device)
    target_net = Net(heavy_model=True).double().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)# optim.RMSprop(policy_net.parameters(), lr=.00025, alpha=.99, eps=1e-6)
    criterion = nn.SmoothL1Loss()# MSELoss()
    env = Atari(device)
    agent = Agent(rm, policy_net, target_net, optimizer, criterion, env, device)
    episode_x = [0]
    reward_y = [0]
    loss_y = [0]
    iterations_x = []
    epsilon_y = []
    if DYNAMIC_PLOT:
        plt.ion()
    fig, (ax1, ax3) = plt.subplots(2)
    fig.set_size_inches(10, 8)
    ax2 = ax1.twinx()
    iterations = 0
    episode = 1
    init_start = None
    while iterations < N_ITERATIONS:
        if init_start == None and iterations >= RM_START_SIZE * K_SKIP_FRAMES:
            init_start = time.time()
        epsilon_y, episode_loss, iterations, episode_reward = agent.run_episode()
        end = time.time()
        episode_x.append(iterations)
        reward_y.append(episode_reward)
        loss_y.append(episode_loss)
        iterations_x.extend(list(range(len(iterations_x) + 1, iterations + 1)))
        if VERBOSE:
            total_estimated_time = ((end - init_start) / iterations) * N_ITERATIONS
            remaining_estimation = 0 if init_start == None else (total_estimated_time - (end - init_start)) / 60
            uptime = 0 if init_start == None else end - init_start
            print(
                'Ep {5} - ite {0}/{1} - reward {2} - eps {3:.4f} - loss {8:.6f} - rm load {4:.2f}% - uptime {6:.2f}m - remaining {7:.2f}m'
                    .format(iterations, N_ITERATIONS, int(round(episode_reward)), epsilon_y[-1], 100. * len(rm) / RM_CAPACITY,
                            episode, uptime, remaining_estimation, episode_loss), flush=True)
        if DYNAMIC_PLOT:
            display_plot(iterations_x, loss_y, epsilon_y, episode_x, reward_y, ax1=ax1, ax2=ax2, ax3=ax3, fig=fig)
        episode += 1
    if DYNAMIC_PLOT:
        plt.ioff()
        plt.show()
    display_plot(iterations_x, loss_y, epsilon_y, episode_x, reward_y, save=True)

