import gym
import random
import torch
import cv2

from config import *

class Atari(object):
    """
    Wrapper used to simplify interactions with the environment
    """

    def __init__(self, device, save_gif = False):
        self._env = gym.make('Breakout-ramNoFrameskip-v4')
        if save_gif:
            self._env = gym.wrappers.Monitor(self._env, "recording", force=True)
        self._state = None
        self._last_lives = 0
        self._device = device
        self._last_k_frames = []

    def reset(self):
        """
        Resets the environment
        Executes a random number of no op actions between 1 and N_NO_OP
        Create the initial state by stacking four identical frames
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
        self._state = torch.tensor(self._last_k_frames, dtype=torch.float64, device=self._device).unsqueeze(0)

    def step(self, action):
        """
        Executes the action in the environment
        Return the pre-processed resulting frame,
        the reward and if the agent lost a life or not

        :param action: Integer, action to perform
        :return: (processed_new_frame, reward, terminal, terminal_life_lost)
        """
        _, reward, terminal, info = self._env.step(action)  # (5★)

        terminal_life_lost = terminal
        if info['ale.lives'] < self._last_lives:
            terminal_life_lost = True
        self._last_lives = info['ale.lives']

        processed_new_frame = self._preprocess_frame(self._env.render(mode='rgb_array'))
        self._last_k_frames.append(processed_new_frame)
        self._last_k_frames.pop(0)
        self._state = torch.tensor(self._last_k_frames, dtype=torch.float64, device=self._device).unsqueeze(0)

        return processed_new_frame, reward, terminal, terminal_life_lost

    def _preprocess_frame(self, f):
        """
        Pre-process one frame

        :param f: Raw frame from the environment
        :return: Grayscaled, cropped and rescaled to 84x84 frame
        """
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = f[34:-18, :]
        f = cv2.resize(f, (84, 84), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        return f

    def get_state(self):
        return self._state

    def render(self, mode = 'human'):
        if mode == 'human':
            self._env.render(mode=mode)
