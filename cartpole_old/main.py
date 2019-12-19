import gym
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

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
        plt.savefig(PLOTS_DIR + FILE_SUFFIX + '_backtb.png')

if __name__ == "__main__":
    rm = ReplayMemory(RM_CAPACITY)
    policy_net = Net(FEATURES_SIZES).double()
    target_net = Net(FEATURES_SIZES).double()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    epsilon_index = 0
    epsilon = INITIAL_EPSILON#EPSILON_VALUES[epsilon_index]
    env = gym.make("CartPole-v1")
    agent = Agent(rm, policy_net, target_net, optimizer, criterion, env)
    reward_mem = [0] * 5
    lr_updated = False
    lr_updated_2 = False
    reward_x = []
    reward_y = []
    epsilon_y = []
    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for episode in range(N_EPISODES):
        total_reward = agent.run_episode(epsilon, update_target_net=(episode % TARGET_UPDATE == 0), save_models = episode == N_EPISODES - 1)
        reward_mem.append(total_reward)
        reward_mem.pop(0)
        reward_x.append(episode)
        reward_y.append(total_reward)
        epsilon_y.append(epsilon)
        display_plot(reward_x, epsilon_y, reward_x, reward_y, ax1=ax1, ax2=ax2, fig=fig)
        #if episode > EPSILON_THRESHOLDS[epsilon_index]:
        #    epsilon_index += 1
        #epsilon = EPSILON_VALUES[epsilon_index]
        epsilon = max(MINIMAL_EPSILON, epsilon - EPSILON_ANNEALING_STEP)
        """
        if epsilon <= EPSILON_THRESHOLD:
            epsilon = max(MINIMAL_EPSILON, epsilon - EPSILON_ANNEALING_STEP_AFTER)
        else:
            epsilon = max(MINIMAL_EPSILON, epsilon - EPSILON_ANNEALING_STEP)
        if not lr_updated and sum(reward_mem) / 5 >= 200:
            lr_updated = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE / 2
        if not lr_updated_2 and sum(reward_mem) / 5 >= 350:
            lr_updated_2 = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE / 4
                """

        """
        if epsilon > MINIMAL_EPSILON:
            if epsilon < EPSILON_THRESHOLD:
                epsilon -= EPSILON_ANNEALING_STEP_AFTER
            else:
                epsilon -= EPSILON_ANNEALING_STEP
                """
    env.close()
    plt.title('Done')
    plt.ioff()
    plt.show()
    display_plot(reward_x, epsilon_y, reward_x, reward_y, save=True)
