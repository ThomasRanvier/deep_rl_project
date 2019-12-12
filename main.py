import gym
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from replay_memory import ReplayMemory
from net import Net

RM_CAPACITY = 100000
N_ENTRIES = 4
N_ACTIONS = 2
FEATURES_SIZES = [N_ENTRIES, 12, 24, 16, 8, N_ACTIONS]
MINIBATCH_SIZE = 32
INITIAL_EPSILON = .5
EPSILON_ANNEALING_STEP = .002
MINIMAL_EPSILON = .075
GAMMA = .999
LEARNING_RATE = .001
N_EPISODES = 400

if __name__ == "__main__":
    rm = ReplayMemory(RM_CAPACITY)
    net = Net(FEATURES_SIZES).double()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    epsilon = INITIAL_EPSILON
    env = gym.make("CartPole-v1")
    reward_hist = [0]
    plt.ion()
    for episode in range(N_EPISODES):
        state = env.reset()
        state = torch.tensor(state).unsqueeze(0)
        terminal = False
        while(not terminal):
            env.render()
            # https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
            # Get output from nn
            output = net(state)
            action_index = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(torch.argmax(output))
            # Initialize action
            action = torch.zeros([N_ACTIONS])
            action[action_index] = 1.
            # Get next state and reward, done is terminal
            state_1, reward, terminal, _ = env.step(action_index)
            reward_hist[episode] += reward
            # Cast all data to same type : unsqueezed tensor
            action = action.unsqueeze(0)
            reward = torch.tensor([reward]).unsqueeze(0)
            state_1 = torch.tensor(state_1).unsqueeze(0)
            # Save transition to replay memory
            rm.push((state, action, reward, state_1, terminal))
            # Sample random minibatch
            minibatch = rm.sample(MINIBATCH_SIZE)
            # Unpack minibatch
            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
            # Get output for the minibatch next states
            output_1_batch = net(state_1_batch)
            # Set y_j to r_j for terminals, otherwise to r_j + GAMMA * max(Q)
            y_batch = torch.cat(tuple(
                reward_batch[i] if minibatch[i][4] else reward_batch[i] + GAMMA * torch.max(output_1_batch[i])
                for i in range(len(minibatch))
            )).double()
            # Extract Q-value
            q_value = torch.sum(net(state_batch) * action_batch, dim=1)

            optimizer.zero_grad()
            # We have to understand why we need this
            y_batch = y_batch.detach()
            # MSE loss
            loss = criterion(q_value, y_batch)
            # Back propagation
            loss.backward()
            optimizer.step()
            # Next state becomes current state
            state = state_1
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.plot(reward_hist)
        reward_hist.append(0)
        if epsilon > MINIMAL_EPSILON:
            epsilon -= EPSILON_ANNEALING_STEP
    env.close()
    plt.title('Done')
    plt.ioff()
    plt.show()
