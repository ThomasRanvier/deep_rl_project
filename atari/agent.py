import torch
import random
import cv2

from config import *

class Agent():
    def __init__(self, rm, policy_net, target_net, optimizer, criterion, env, device):
        self._rm = rm
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._env = env
        self._last_action = random.randint(0, N_ACTIONS - 1)
        self._epsilon = INITIAL_EPSILON
        self._iteration = 0
        self._loss_hist = []
        self._epsilon_hist = []
        self._device = device
        self._current_state_id = -K_SKIP_FRAMES# Start counter at -K_SKIP_FRAMES so that state 0 corresponds to the first K frames of the memory

    def _optimize_model(self):
        # Sample random minibatch, it is already unpacked and ready to use
        state_batch, action_batch, reward_batch, state_1_batch, terminal_batch = self._rm.sample(MINIBATCH_SIZE)

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
            reward_batch[i] if terminal_batch[i] else reward_batch[i] + GAMMA * torch.max(output_1_batch[i])
            for i in range(len(terminal_batch))
        )).double()

        self._optimizer.zero_grad()
        # Detach so that we do not minimize the target net
        next_q_values = next_q_values.detach()
        # False
        loss = self._criterion(q_values, next_q_values)
        self._loss_hist.append(loss.item())
        # Back propagation
        loss.backward()

        # del variables just in case
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

        # Update epsilon every params update (starts updating after RM_START_SIZE)
        eps_1 = max(MINIMAL_EPSILON_1, self._epsilon - EPSILON_ANNEALING_STEP_1)
        if eps_1 < self._epsilon:
            self._epsilon = eps_1
        else:
            self._epsilon = max(MINIMAL_EPSILON_2, self._epsilon - EPSILON_ANNEALING_STEP_2)

    def _increment_iteration(self):
        self._iteration += 1
        self._epsilon_hist.append(self._epsilon)

    def _no_op(self):
        # Perform N times the no op action
        # Those N iterations are not part of the learning process (counter not incremented, epsilon unchanged, etc.)
        for _ in range(random.randint(1, N_NO_OP)):
            # Play no op action
            _, _, _, _ = self._env.step(NO_OP_ACTION)
        # Once no op actions are done we get the current frame and add it K times to the rm
        # Which means that each episode we 'skip' K states
        self._get_frame(n_frames=K_SKIP_FRAMES)

    def _preprocess_frame(self, f):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[34:-18, :]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        return f

    """
    Get and preprocess one frame from the game
    Also, add it to the replay memory
    Each frame means new state, so we update current_state_id
    """
    def _get_frame(self, n_frames = 1):
        f = self._preprocess_frame(self._env.render(mode='rgb_array'))
        for _ in range(n_frames):
            self._rm.add_frame(f)
        self._current_state_id += n_frames

    def run_episode(self):
        self._env.reset()
        self._no_op()
        episode_reward = 0
        # Request the last state to the rm
        state = self._rm.get_state(self._current_state_id)
        # Initialize episode variables
        terminal = False
        n_frames = 0
        lives = 5
        self._loss_hist = []
        while not terminal:
            if DISPLAY_SCREEN:
                self._env.render(mode='human')
            # Select new action every k frames
            if n_frames % K_SKIP_FRAMES == 0:
                if random.random() < self._epsilon:
                    # Random action
                    self._last_action = random.randint(0, N_ACTIONS - 1)
                else:
                    # Get output from nn applied on last k preprocessed frames
                    with torch.no_grad():
                        output = self._policy_net(state)
                    self._last_action = int(torch.argmax(output))

            # Play the selected action
            _, reward, terminal, obs = self._env.step(self._last_action + 1)
            episode_reward += reward
            # Used to notify the agent that it lost a life in this state
            rm_terminal = terminal
            current_lives = obs['ale.lives']
            if current_lives < lives or terminal:
                rm_terminal = True
                lives = current_lives
            # Clipping the reward, even if I don't really see when the reward would be more than 1 or less than 0...
            reward = 1 if reward > 0 else (-1 if reward < 0 else 0)

            # Get resulting frame
            self._get_frame()

            # Save transition to replay memory
            self._rm.push(self._current_state_id - 1, self._last_action, reward, self._current_state_id, rm_terminal)

            # Next state becomes current state
            state = self._rm.get_state(self._current_state_id)

            # Optimize the nn every k frames
            if n_frames % K_SKIP_FRAMES == K_SKIP_FRAMES - 1:
                # Start params update once RM is populated enough
                if len(self._rm) >= RM_START_SIZE:
                    self._optimize_model()
                # increment total iterations count and epsilon, once every 4 frames
                self._increment_iteration()
            # Increment episode iteration count to know when to select a new action
            n_frames += 1
        if self._iteration >= N_ITERATIONS:
            torch.save(self._policy_net, MODELS_DIR + 'policy_net_' + FILE_SUFFIX + '_c' + str(N_ITERATIONS) + '.pt')
        return self._epsilon_hist, sum(self._loss_hist) / max(1., len(self._loss_hist)), self._iteration, episode_reward
