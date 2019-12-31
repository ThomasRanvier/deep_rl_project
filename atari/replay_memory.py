import numpy as np
import random
import torch

from config import *

class ReplayMemory(object):
    def __init__(self, device):
        self.size = RM_CAPACITY
        self.frame_height = 84
        self.frame_width = 84
        self.agent_history_length = K_SKIP_FRAMES
        self.batch_size = MINIBATCH_SIZE
        self.count = 0
        self.current = 0
        self._device = device

        # Pre-allocate memory
        self.actions = np.empty([self.size, N_ACTIONS], dtype=np.int32)
        self.rewards = np.empty([self.size, 1], dtype=np.float32)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError('Dimension of frame is wrong!')
        act = [0.] * N_ACTIONS
        act[action] = 1.
        self.actions[self.current, ...] = act
        self.frames[self.current, ...] = frame
        self.rewards[self.current, 0] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        if self.count is 0:
            raise ValueError('The replay memory is empty!')
        if index < self.agent_history_length - 1:
            raise ValueError('Index must be min ' + str(self.agent_history_length - 1))
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self):
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)

        return torch.from_numpy(self.states).double().to(self._device), \
               torch.from_numpy(self.actions[self.indices]).double().to(self._device), \
               torch.from_numpy(self.rewards[self.indices]).double().to(self._device), \
               torch.from_numpy(self.new_states).double().to(self._device),\
               self.terminal_flags[self.indices].tolist()
