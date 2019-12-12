import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make("CartPole-v1")
observation = env.reset()
reward_hist = [0]
episode = 0
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    reward_hist[episode] += 1
    if done:
        episode += 1
        reward_hist.append(0)
        observation = env.reset()
env.close()
plt.plot(reward_hist)
plt.show()
