import torch
import random

from config import *

class Agent():
    def __init__(self, rm, policy_net, target_net, optimizer, criterion, env, device):
        self._rm = rm
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._env = env
        self._device = device
        self._epsilon = INITIAL_EPSILON
        self._iteration = 0
        self._loss_hist = []
        self._epsilon_hist = []

    def _optimize_model(self):
        self._loss_hist.append(0)
        # Sample random minibatch
        minibatch = self._rm.sample(MINIBATCH_SIZE)
        # Unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        # Extract Q-values for the current states from the minibatch
        # Q(s, a)
        # We multiply the output by the actions vectors as a mask
        # ex: output = [.25, .75] * action = [1., 0.] = [.25, 0.]
        # Then we sum it on dimension 1
        # ex: sum([.25, 0.]) = .25
        # We then obtain a tensor that contains one sum for each state in the batch
        q_values = torch.sum(self._policy_net(state_batch) * action_batch, dim=1)

        # Extract Q-values for the minibatch next states
        # ^Q(s', a)
        output_1_batch = self._target_net(state_1_batch)
        # Set y_j to r_j for terminals, otherwise to r_j + GAMMA * max(Q)
        next_q_values = torch.cat(tuple(
            reward_batch[i] if minibatch[i][4] else reward_batch[i] + GAMMA * torch.max(output_1_batch[i])
            for i in range(len(minibatch))
        )).double()

        self._optimizer.zero_grad()
        # Detach so that we do not minimize the target net
        next_q_values = next_q_values.detach()
        # MSE loss
        loss = self._criterion(q_values, next_q_values)
        self._loss_hist.append(float(loss))
        # Back propagation
        loss.backward()
        # DQN gradient clipping
        # https://stackoverflow.com/questions/47036246/dqn-q-loss-not-converging
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

        # Update target net if needed
        if self._iteration % TARGET_UPDATE == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())
        # Save models if needed
        if self._iteration in SAVE_MODELS:
            torch.save(self._policy_net, MODELS_DIR + 'policy_net_' + FILE_SUFFIX + '_c' + str(self._iteration) + '.pt')

    def _preprocess_state(self, state):
        state = torch.tensor(state, dtype=torch.float64, device=self._device).unsqueeze(0)
        return state

    def _increment_iteration(self):
        self._iteration += 1
        self._epsilon_hist.append(self._epsilon)
        self._epsilon = max(MINIMAL_EPSILON, self._epsilon - EPSILON_ANNEALING_STEP)

    def run_episode(self):
        state = self._env.reset()
        state = self._preprocess_state(state)
        episode_reward = 0
        terminal = False
        self._loss_hist = []
        while not terminal:
            if DISPLAY_SCREEN:
                self._env.render(mode='human')
            # Get output from nn
            if random.random() < self._epsilon:
                # Random action
                action_index = random.randint(0, N_ACTIONS - 1)
            else:
                # Get output from nn applied on last state
                with torch.no_grad():
                    output = self._policy_net(state)
                action_index = int(torch.argmax(output))

            # Get next state and reward, done is terminal
            state_1, reward, terminal, _ = self._env.step(action_index)
            # Increment iteration counter and update epsilon, etc.
            self._increment_iteration()
            episode_reward += reward
            
            # Scale the reward depending on the distance of the cart from the center
            if reward > 0:
                reward -= abs(state_1[0]) / 4.8

            # Cast all data to same type : unsqueezed tensor
            action = torch.zeros([N_ACTIONS], device=self._device)
            action[action_index] = 1.
            action = action.unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float64, device=self._device).unsqueeze(0)
            state_1 = self._preprocess_state(state_1)
            # Save transition to replay memory
            self._rm.push((state, action, reward, state_1, terminal))
            # Next state becomes current state
            state = state_1
            # Optimize the nn
            self._optimize_model()
        if self._iteration >= N_ITERATIONS:
            torch.save(self._policy_net, MODELS_DIR + 'policy_net_' + FILE_SUFFIX + '_c' + str(N_ITERATIONS) + '.pt')
        return self._epsilon_hist, sum(self._loss_hist) / len(self._loss_hist), self._iteration, episode_reward
