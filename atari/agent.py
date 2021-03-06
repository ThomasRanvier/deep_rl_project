import torch
import random

from config import *

class Agent():
    """
    The DQN agent, implementation of the DQN algorithm
    """

    def __init__(self, rm, policy_net, target_net, optimizer, criterion, env, device):
        self._rm = rm
        self._policy_net = policy_net
        self._target_net = target_net
        self._optimizer = optimizer
        self._criterion = criterion
        self._env = env
        self._last_action = random.randint(0, N_ACTIONS - 1)
        self._current_epsilon_id = 0
        self._epsilon = EPS_VALUES[self._current_epsilon_id]
        self._iteration = 0
        self._loss_hist = []
        self._epsilon_hist = []
        self._device = device
        self._batch_indices = list(range(MINIBATCH_SIZE))

    def _optimize_model(self):
        """
        Perform one params update
        Request a minibatch from the replay memory
        Get the q-values of the states and action from the policy network
        Get the q-values of the next states from the target network
        Performs a back-propagation and an optimizer step
        The target network is updated and the current model is saved if needed
        The epsilon value is updated
        """
        # Sample random minibatch, it is already unpacked and ready to use
        state_batch, action_batch, reward_batch, state_1_batch, terminal_batch = self._rm.get_minibatch()

        # Extract Q-values for the current states from the minibatch
        # Q(s, a)
        # We get an output for the state and all possible actions from the policy net
        # Then we select only the q-values corresponding to the performed action
        q_values = self._policy_net(state_batch)
        q_values = q_values[self._batch_indices, action_batch.tolist()]

        # Extract Q-values for the minibatch next states
        # ^Q(s', a)
        # Detach so that we do not minimize the target net
        next_q_values = self._target_net(state_1_batch).detach().double()
        # Double DQN, https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/DQN_agent.py
        best_actions_indices = torch.argmax(self._policy_net(state_1_batch), dim=-1)
        # Select the actions depending on the argmax performed on policy net output
        next_q_values = next_q_values[self._batch_indices, best_actions_indices]
        # Set y_j to r_j for terminals, otherwise to r_j + GAMMA * max(Q)
        next_q_values = reward_batch + (GAMMA * next_q_values * (1 - terminal_batch))

        self._optimizer.zero_grad()
        # q_values has grad activated, next_q_values doesn't, so only _policy_net will be optimized
        loss = self._criterion(q_values, next_q_values)
        self._loss_hist.append(loss.item())
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

        # Update epsilon every params update (starts updating after RM_START_SIZE)
        if self._current_epsilon_id < len(EPS_TRIGGERS):
            if self._iteration == EPS_TRIGGERS[self._current_epsilon_id]:
                self._current_epsilon_id += 1
        self._epsilon = self._epsilon + EPS_STEPS[self._current_epsilon_id]

    def _increment_iteration(self):
        self._iteration += 1
        self._epsilon_hist.append(self._epsilon)

    def run_episode(self):
        """
        Run one epiode
        """
        self._env.reset()
        episode_reward = 0
        # Initialize episode variables
        terminal = False
        terminal_life_lost = False
        n_frames = 0
        self._loss_hist = []
        while not terminal:
            if DISPLAY_SCREEN:
                self._env.render(mode='human')
            # Select new action every k frames
            #if n_frames % K_SKIP_FRAMES == 0:
            if random.random() < self._epsilon:
                # Random action
                self._last_action = random.randint(0, N_ACTIONS - 1)
            else:
                # Get output from nn applied on last k preprocessed frames
                with torch.no_grad():
                    output = self._policy_net(self._env.get_state())
                self._last_action = int(torch.argmax(output))

            played_action = 0 if terminal_life_lost else self._last_action
            # Play the selected action, fire if just lost a life
            processed_new_frame, reward, terminal, terminal_life_lost = self._env.step(played_action + 1)
            episode_reward += reward
            # Clipping the reward
            clipped_reward = 1 if reward > 0 else (-1 if reward < 0 else 0)

            # Save transition to replay memory
            self._rm.add_experience(played_action, processed_new_frame, clipped_reward, terminal_life_lost)

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
