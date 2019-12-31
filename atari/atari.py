import gym
import random
import torch
import cv2

from config import *

class Atari(object):
    """Wrapper for the environment provided by gym"""

    def __init__(self, device):
        self._env = gym.make('Breakout-ramNoFrameskip-v4')
        self.state = None
        self._last_lives = 0
        self._device = device
        self._last_k_frames = []

    def reset(self):
        """
        Resets the environment and stacks four frames ontop of each other to
        create the first state
        """
        self._env.reset()
        self._last_lives = 0
        # Perform N times the no op action
        # Those N iterations are not part of the learning process (counter not incremented, epsilon unchanged, etc.)
        for _ in range(random.randint(1, N_NO_OP)):
            _, _, _, _ = self._env.step(NO_OP_ACTION)
        processed_frame = self._preprocess_frame(self._env.render(mode='rgb_array'))
        self._last_k_frames = []
        for _ in range(K_SKIP_FRAMES):
            self._last_k_frames.append(processed_frame)
        # Create the initial state with four identical frames
        self.state = torch.tensor(self._last_k_frames, dtype=torch.float64, device=self._device).unsqueeze(0)

    def step(self, action):
        """
        Args:
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        _, reward, terminal, info = self._env.step(action)  # (5★)

        terminal_life_lost = terminal
        if info['ale.lives'] < self._last_lives:
            terminal_life_lost = True
        self._last_lives = info['ale.lives']

        processed_new_frame = self._preprocess_frame(self._env.render(mode='rgb_array'))
        self._last_k_frames.append(processed_new_frame)
        self._last_k_frames.pop(0)
        self.state = torch.tensor(self._last_k_frames, dtype=torch.float64, device=self._device).unsqueeze(0)

        return processed_new_frame, reward, terminal, terminal_life_lost

    def _preprocess_frame(self, f):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[34:-18, :]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        return f

