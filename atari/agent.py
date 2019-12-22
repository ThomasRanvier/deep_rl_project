import torch
import random
import cv2

from config import *

class Agent():
    def __init__(self, rm, policy_net, target_net, optimizer, criterion, env):
        self._rm = rm
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._env = env
        self._last_action = random.randint(0, N_ACTIONS - 1)
        self._last_k_frames = []
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
        # loss
        loss = self._criterion(q_values, next_q_values)
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

    def _preprocess_state(self, last_k_frames):
        stacked_frames = []
        for f in last_k_frames:
            f = f[:, :, 0] * 0.299 + f[:, :, 1] * 0.587 + f[:, :, 2] * 0.114
            f = f[58:-18, 8:-8]
            f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
            stacked_frames.append(f)
            #cv2.imshow('image', f)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        state = stacked_frames
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        return state

    def _get_last_k_frames(self):
        iteration_reward = 0
        last_k_frames = []
        for i in range(K_SKIP_FRAMES):
            if DISPLAY_SCREEN:
                # Display screen
                self._env.render(mode='human')
            # Get the frame
            img = self._env.render(mode='rgb_array')
            last_k_frames.append(img)
            # Play last chosen action
            _, reward, terminal, _ = self._env.step(self._last_action + 1)
            iteration_reward += reward
            if terminal:
                # If terminal we complete the state with the previous frames
                # Otherwise the state might include less than 4 frames, the CNN requires 4 frames to work
                for j in range((K_SKIP_FRAMES - 1) - i):
                    index = -j - 1
                    last_k_frames.insert(0, self._last_k_frames[index])
                iteration_reward = -1
                break
        self._last_k_frames = last_k_frames
        self._increment_iteration()
        return (iteration_reward, last_k_frames, terminal)

    def _increment_iteration(self):
        self._iteration += 1
        self._epsilon_hist.append(self._epsilon)
        self._epsilon = max(MINIMAL_EPSILON, self._epsilon - EPSILON_ANNEALING_STEP)

    def run_episode(self):
        self._env.reset()
        # Get the last k frames with the cumulated reward and terminal bool
        episode_reward, last_k_frames, terminal = self._get_last_k_frames()
        # Preprocess the k last frames to get one state
        state = self._preprocess_state(last_k_frames)
        terminal = False
        self._loss_hist = []
        while not terminal:
            if random.random() < self._epsilon:
                # Random action
                self._last_action = random.randint(0, N_ACTIONS - 1)
            else:
                # Get output from nn applied on last k preprocessed frames
                with torch.no_grad():
                    output = self._policy_net(state)
                self._last_action = int(torch.argmax(output))

            # Initialize action
            action = torch.zeros([N_ACTIONS])
            action[self._last_action] = 1.

            # Get the last k frames with the cumulated reward and terminal bool
            iteration_reward, last_k_frames, terminal = self._get_last_k_frames()
            episode_reward += 0 if iteration_reward < 0 else iteration_reward
            # Preprocess the k last frames to get one state
            state_1 = self._preprocess_state(last_k_frames)

            # Cast all data to same type : unsqueezed tensor
            action = action.unsqueeze(0)
            iteration_reward = torch.tensor([iteration_reward], dtype=torch.float64).unsqueeze(0)
            # Save transition to replay memory
            self._rm.push((state, action, iteration_reward, state_1, terminal))
            # Next state becomes current state
            state = state_1
            # Optimize the nn
            self._optimize_model()
        if self._iteration >= N_ITERATIONS:
            torch.save(self._policy_net, MODELS_DIR + 'policy_net_' + FILE_SUFFIX + '_c' + str(N_ITERATIONS) + '.pt')
        return self._epsilon_hist, sum(self._loss_hist) / len(self._loss_hist), self._iteration, episode_reward
