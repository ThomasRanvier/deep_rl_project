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
        self._last_k_frames_indices = []
        self._epsilon = INITIAL_EPSILON
        self._iteration = 0
        self._loss_hist = []
        self._epsilon_hist = []

    def _optimize_model(self):
        # Sample random minibatch
        minibatch = self._rm.sample(MINIBATCH_SIZE)
        # Unpack minibatch
        state_batch = torch.cat(tuple(self._get_state_from_indices(d[0]) for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(self._get_state_from_indices(d[3]) for d in minibatch))

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
        # False
        loss = self._criterion(q_values, next_q_values)
        self._loss_hist.append(loss.item())
        # Back propagation
        loss.backward()

        # del variables to avoid having an increasing memory consumption during the whole training
        del loss
        del next_q_values
        del output_1_batch
        del q_values

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

    def _increment_iteration(self):
        self._iteration += 1
        self._epsilon_hist.append(self._epsilon)
        self._epsilon = max(MINIMAL_EPSILON, self._epsilon - EPSILON_ANNEALING_STEP)

    def _no_op(self):
        # Perform N times the no op action
        # Those N iterations are not part of the learning process (counter not incremented, epsilon unchanged, etc.)
        self._last_k_frames_indices = []
        for _ in range(N_NO_OP):
            # Play no op action
            _, _, _, _ = self._env.step(NO_OP_ACTION)
            # Get resulting frame
            self._get_frame()

    def _preprocess_frame(self, f):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[58:-18, 8:-8]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        #cv2.imshow('image', f)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return f

    """
    Get and preprocess one frame from the game
    Also, add it to the replay memory
    """
    def _get_frame(self):
        f = self._preprocess_frame(self._env.render(mode='rgb_array'))
        f_idx = self._rm.add_frame(f)
        self._last_k_frames_indices.append(f_idx)
        if len(self._last_k_frames_indices) > K_SKIP_FRAMES:
            self._last_k_frames_indices.pop(0)

    """
    Return a state: 4 frames of 84x84
    """
    def _get_state_from_indices(self, f_indices):
        state = []
        for f_idx in f_indices:
            state.append(self._rm.get_frame(f_idx))
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        return state

    def run_episode(self):
        self._env.reset()
        self._no_op()
        episode_reward = 0
        # Preprocess the k last frames to get one state
        state = self._get_state_from_indices(self._last_k_frames_indices)
        state_frames_indices = [f_idx for f_idx in self._last_k_frames_indices]
        # Initialize episode variables
        terminal = False
        episode_iteration = 0
        lives = 5
        self._loss_hist = []
        while not terminal:
            if DISPLAY_SCREEN:
                self._env.render(mode='human')
            # Select new action every k frames
            if episode_iteration % K_SKIP_FRAMES == 0:
                if random.random() < self._epsilon:
                    # Random action
                    self._last_action = random.randint(0, N_ACTIONS - 1)
                else:
                    # Get output from nn applied on last k preprocessed frames
                    with torch.no_grad():
                        output = self._policy_net(state)
                    self._last_action = int(torch.argmax(output))

            # Play the selected action
            _, reward, terminal, obs = self._env.step(self._last_action)
            episode_reward += reward
            current_lives = obs['ale.lives']

            if current_lives < lives or terminal:
                reward = -1

            # Get resulting frame
            self._get_frame()

            # Get the new state
            state_1 = self._get_state_from_indices(self._last_k_frames_indices)
            state_1_frames_indices = [f_idx for f_idx in self._last_k_frames_indices]
            # Cast all data to same type : unsqueezed tensor
            action = torch.zeros([N_ACTIONS])
            action[self._last_action] = 1.
            action = action.unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float64).unsqueeze(0)
            # Save transition to replay memory
            self._rm.push((state_frames_indices, action, reward, state_1_frames_indices, terminal))

            # Next state becomes current state
            state = state_1
            state_frames_indices = state_1_frames_indices
            # Optimize the nn every k frames
            if episode_iteration % K_SKIP_FRAMES == K_SKIP_FRAMES - 1:
                self._optimize_model()
                # increment total iterations count and epsilon, once every 4 frames
                self._increment_iteration()
            # increment episode iteration count to know when to select a new action
            episode_iteration += 1
            if current_lives < lives and not terminal:
                lives = current_lives
                self._no_op()
        if self._iteration >= N_ITERATIONS:
            torch.save(self._policy_net, MODELS_DIR + 'policy_net_' + FILE_SUFFIX + '_c' + str(N_ITERATIONS) + '.pt')
        return self._epsilon_hist, sum(self._loss_hist) / len(self._loss_hist), self._iteration, episode_reward
