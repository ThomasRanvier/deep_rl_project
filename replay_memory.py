import random

class ReplayMemory():
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = []

    def push(self, transition):
        if len(self._memory) >= self._capacity:
            self._memory.pop(0)
        self._memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self._memory, min(len(self._memory), batch_size))

    def __len__(self):
        return len(self._memory)
