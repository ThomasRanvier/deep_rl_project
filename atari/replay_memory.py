import numpy as np
import random
import torch

from config import *

class ReplayMemory(object):
    """
    Replay memory, used to store the transitions and return the minibatches
    Hugely inspired from https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
    """

    def __init__(self, device):
        self._count = 0
        self._current_id = 0
        self._device = device

        # Variables used to store the transitions
        self._actions = np.empty(RM_CAPACITY, dtype=np.uint8)
        self._rewards = np.empty(RM_CAPACITY, dtype=np.float32)
        self._frames = np.empty((RM_CAPACITY, 84, 84), dtype=np.uint8)
        self._terminal_flags = np.empty(RM_CAPACITY, dtype=np.bool)

        # Variables used to return the minibatches
        self._states = np.empty((MINIBATCH_SIZE, K_SKIP_FRAMES, 84, 84), dtype=np.float32)
        self._next_states = np.empty((MINIBATCH_SIZE, K_SKIP_FRAMES, 84, 84), dtype=np.float32)
        self._indices = np.empty(MINIBATCH_SIZE, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        """
        Add an experience to the memory
        The action is transformed to a bool array before storing it, because it is used as a mask later on

        :param action: The action performed
        :param frame: The resulting pre-processed frame
        :param reward: The resulting reward
        :param terminal: If the state was final or not
        """
        self._actions[self._current_id] = action
        self._frames[self._current_id, ...] = frame
        self._rewards[self._current_id] = reward
        self._terminal_flags[self._current_id] = terminal
        self._count = max(self._count, self._current_id + 1)
        self._current_id = (self._current_id + 1) % RM_CAPACITY

    def _get_state(self, index):
        """
        Return the corresponding state from the memory, a state is composed from four frames
        Normalize the frames at that moment to save memory, since the frames are stored as np.uint8

        :param index: Index of the state
        :return: The four frames of the state
        """
        if index < K_SKIP_FRAMES - 1:
            raise ValueError('Index must be min ' + str(K_SKIP_FRAMES - 1))
        state = self._frames[index - K_SKIP_FRAMES + 1:index + 1, ...]
        state = state / 255.
        return state

    def _get_valid_indices(self):
        """
        Fills the indices array with transition valid indices
        A valid index is one that is >= to K_SKIP_FRAMES: because otherwise we cannot return a full state with k frames
        A valid index must not select a state that is being currently replaced by a new one
        A valid index must not select a state with a terminal flag in the first k - 1 frames
        """
        for i in range(MINIBATCH_SIZE):
            while True:
                index = random.randint(K_SKIP_FRAMES, self._count - 1)
                if index < K_SKIP_FRAMES:
                    continue
                if index >= self._current_id >= index - K_SKIP_FRAMES:
                    continue
                if self._terminal_flags[index - K_SKIP_FRAMES:index].any():
                    continue
                break
            self._indices[i] = index

    def get_minibatch(self):
        """
        Give a minibatch

        :return: A tuple with five variables:
            - A double torch tensor on the selected device that contains the states
            - A double torch tensor on the selected device that contains the actions
            - A double torch tensor on the selected device that contains the rewards
            - A double torch tensor on the selected device that contains the next states
            - A double torch tensor on the selected device that contains the terminal flags
        """
        if self._count < K_SKIP_FRAMES:
            raise ValueError('Not enough memories to get a minibatch')

        self._get_valid_indices()

        for i, idx in enumerate(self._indices):
            self._states[i] = self._get_state(idx - 1)
            self._next_states[i] = self._get_state(idx)

        # Cast to double tensors on selected device before returning
        return torch.from_numpy(self._states).double().to(self._device), \
               torch.from_numpy(self._actions[self._indices]).to(self._device), \
               torch.from_numpy(self._rewards[self._indices]).double().to(self._device), \
               torch.from_numpy(self._next_states).double().to(self._device),\
               torch.from_numpy(self._terminal_flags[self._indices]).double().to(self._device)

    def __len__(self):
        return self._count