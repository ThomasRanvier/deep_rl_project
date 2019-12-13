import gym
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn

from replay_memory import ReplayMemory
from net import Net
from agent import Agent
from config import *

if __name__ == "__main__":
    rm = ReplayMemory(RM_CAPACITY)
    policy_net = Net(FEATURES_SIZES).double()
    target_net = Net(FEATURES_SIZES).double()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    epsilon = INITIAL_EPSILON
    env = gym.make("Breakout-ramNoFrameskip-v4")
    agent = Agent(rm, policy_net, target_net, optimizer, criterion, env)
    reward_hist = []
    plt.ion()
    for episode in range(N_EPISODES):
        total_reward = agent.run_episode(epsilon, update_target_net=(episode % TARGET_UPDATE == 0))
        reward_hist.append(total_reward)
        plt.figure(1)
        plt.clf()
        plt.title('Training... eps: ' + str(round(epsilon, 2)))
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(reward_hist)
        epsilon -= (epsilon > MINIMAL_EPSILON) * EPSILON_ANNEALING_STEP
    env.close()
    plt.title('Done')
    plt.ioff()
    plt.show()
