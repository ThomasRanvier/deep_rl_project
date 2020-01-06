N_ACTIONS = 3
K_SKIP_FRAMES = 4
MINIBATCH_SIZE = 1#32# Useless to go above 32
N_ITERATIONS = 2750000
RM_CAPACITY = 1000000
RM_START_SIZE = 500000
EPS_VALUES = [0., 0., .25, .01, .01]
EPS_STEPS = [0., (.25 - 0.) / 1000000., (.1 - .25) / 1000000., 0.]# TODO compute automaticaly using EPS_VALUES
EPS_TRIGGERS = [500000, 1500000, 2500000]
GAMMA = .99# .99 seems to give good maybe even better results than .999
LEARNING_RATE = .00000625#.0000625# https://github.com/dennybritz/reinforcement-learning/issues/30#issuecomment-407910672# .0000625# In a later DeepMind paper called "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al. 2017 RMSProp was substituted for Adam with a learning rate of 0.0000625
TARGET_UPDATE = 2500# 2500 car 10000/4: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756,  the target network update frequency is, thus, measured in numSteps and not in parameter updates. Which means updated every 10000/4 iterations !
DISPLAY_SCREEN = False
DYNAMIC_PLOT = False
VERBOSE = True
PLOTS_DIR = 'reward_plots/'
MODELS_DIR = 'saved_models/'
FILE_SUFFIX = 'lr' + str(LEARNING_RATE) + '_bs' + str(MINIBATCH_SIZE) + '_tu' + str(TARGET_UPDATE) + '_it' + \
              str(N_ITERATIONS) + '_g' + str(GAMMA)
SAVE_EVERY = 50000
SAVE_MODELS = [i for i in range(50000, N_ITERATIONS - SAVE_EVERY + 1, SAVE_EVERY)]
N_NO_OP = 20# The agent performs between 1 and N_NO_OP NO_OP_ACTIONs at the start of each episode, point being to start in different situations
NO_OP_ACTION = 1# We choose the fire action so that the ball is launched, once in the game it does nothing anymore
# https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
