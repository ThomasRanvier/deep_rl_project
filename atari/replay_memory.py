import random
import torch

from config import *

class ReplayMemory():
    def __init__(self, capacity, device):
        self._capacity = capacity
        self._device = device
        self._memory = []
        self._frames = []

    def push(self, state_id, chosen_action, reward, next_state_id, terminal):
        # Cast all data to same type : unsqueezed tensor
        action = torch.zeros([N_ACTIONS], device=self._device)
        action[chosen_action] = 1.
        action = action.unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float64, device=self._device).unsqueeze(0)
        # Initialize the transition tuple
        transition = (state_id, action, reward, next_state_id, terminal)
        # Push it to the memory
        if len(self._memory) >= self._capacity:
            self._memory.pop(0)
        self._memory.append(transition)

    def sample(self, batch_size):
        # Sample random minibatch
        minibatch = random.sample(self._memory, min(len(self._memory), batch_size))
        # Unpack minibatch
        state_batch = torch.cat(tuple(self.get_state(d[0]) for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(self.get_state(d[3]) for d in minibatch))
        terminal_batch = tuple(d[4] for d in minibatch)
        return state_batch, action_batch, reward_batch, state_1_batch, terminal_batch

    def add_frame(self, frame):
        # Push the last frame to the frames memory
        if len(self._frames) >= self._capacity + K_SKIP_FRAMES:
            self._frames.pop(0)
        self._frames.append(frame)

    def get_state(self, state_id):
        # Extract the K frames corresponding to the state_id
        frames = []
        for i in range(state_id, state_id + K_SKIP_FRAMES):
            frames.append(self._frames[i % (self._capacity + K_SKIP_FRAMES)])
        # Cast the K frames to a tensor
        state = torch.tensor(frames, dtype=torch.float64, device=self._device).unsqueeze(0)
        return state

    def __len__(self):
        return len(self._memory)
