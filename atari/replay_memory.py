import random

class ReplayMemory():
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = []
        self._frames = []

    def push(self, transition):
        if len(self._memory) >= self._capacity:
            self._memory.pop(0)
        self._memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self._memory, min(len(self._memory), batch_size))

    def add_frame(self, frame):
        f_idx = None
        i = 0
        for f in self._frames:
            identical = True
            for i in range(len(f)):
                if f[i] != frame[i]:
                    identical = False
                    break
            if identical:
                f_idx = i
                break
            i += 1
        if f_idx is None:
            f_idx = len(self._frames)
            self._frames.append(frame)
        return f_idx

    def get_frame(self, f_idx):
        return self._frames[f_idx]

    def __len__(self):
        return len(self._memory)
