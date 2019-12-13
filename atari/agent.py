import torch
import random

from config import *

class Agent():
    def __init__(self, rm, policy_net, target_net, optimizer, criterion, env):
        self._rm = rm
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._env = env

    def _optimize_model(self):
        # Sample random minibatch
        minibatch = self._rm.sample(MINIBATCH_SIZE)
        # Unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        # Extract Q-values for the current states from the minibatch
        q_values = torch.sum(self._policy_net(state_batch) * action_batch, dim=1)

        # Extract Q-values for the minibatch next states
        output_1_batch = self._target_net(state_1_batch)
        # Set y_j to r_j for terminals, otherwise to r_j + GAMMA * max(Q)
        next_q_values = torch.cat(tuple(
            reward_batch[i] if minibatch[i][4] else reward_batch[i] + GAMMA * torch.max(output_1_batch[i])
            for i in range(len(minibatch))
        )).double()

        self._optimizer.zero_grad()
        # We have to understand why we need this
        next_q_values = next_q_values.detach()
        # MSE loss
        loss = self._criterion(q_values, next_q_values)
        # Back propagation
        loss.backward()
        self._optimizer.step()

    def run_episode(self, epsilon, update_target_net = False):
        state = self._env.reset()
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        total_reward = 0
        terminal = False
        while(not terminal):
            self._env.render()
            # https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
            # Get output from nn
            with torch.no_grad():
                output = self._policy_net(state)
            action_index = random.randint(0, N_ACTIONS - 1) if random.random() < epsilon else int(torch.argmax(output))
            # Initialize action
            action = torch.zeros([N_ACTIONS])
            action[action_index] = 1.
            # Get next state and reward, done is terminal
            state_1, reward, terminal, _ = self._env.step(action_index)
            total_reward += reward
            # Cast all data to same type : unsqueezed tensor
            action = action.unsqueeze(0)
            reward = torch.tensor([reward]).unsqueeze(0)
            state_1 = torch.tensor(state_1, dtype=torch.float64).unsqueeze(0)
            # Save transition to replay memory
            self._rm.push((state, action, reward, state_1, terminal))
            # Next state becomes current state
            state = state_1
            # Optimize the nn
            self._optimize_model()
            #self._target_net.load_state_dict(self._policy_net.state_dict())
        if update_target_net:
            self._target_net.load_state_dict(self._policy_net.state_dict())
        return total_reward
