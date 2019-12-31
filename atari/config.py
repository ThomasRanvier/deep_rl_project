N_ACTIONS = 3
K_SKIP_FRAMES = 4
MINIBATCH_SIZE = 32
N_ITERATIONS = 3500000
RM_CAPACITY = 1000000
RM_START_SIZE = 50000
# Implementing the epsilon annealing steps 1 and 2, DeepMind then keeps epsilon=.1 after annealing, however, we chose to decrease it to epsilon=.01 over the remaining frames as suggested by the OpenAi Baselines for DQN
INITIAL_EPSILON = 1.
MINIMAL_EPSILON_1 = .1
MINIMAL_EPSILON_2 = .01
EPS_DECAY_STEPS_1 = 1500000
EPS_DECAY_FRACTION_1 = EPS_DECAY_STEPS_1 / N_ITERATIONS
EPSILON_ANNEALING_STEP_1 = (INITIAL_EPSILON - MINIMAL_EPSILON_1) / (EPS_DECAY_FRACTION_1 * N_ITERATIONS)
EPS_DECAY_STEPS_2 = N_ITERATIONS - EPS_DECAY_STEPS_1 - (RM_START_SIZE // K_SKIP_FRAMES)
EPS_DECAY_FRACTION_2 = EPS_DECAY_STEPS_2 / N_ITERATIONS
EPSILON_ANNEALING_STEP_2 = (MINIMAL_EPSILON_1 - MINIMAL_EPSILON_2) / (EPS_DECAY_FRACTION_2 * N_ITERATIONS)
GAMMA = .999
LEARNING_RATE = .0000625# In a later DeepMind paper called "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al. 2017 RMSProp was substituted for Adam with a learning rate of 0.0000625
TARGET_UPDATE = 2500# 2500 car 10000/4: https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756,  the target network update frequency is, thus, measured in numSteps and not in parameter updates. Which means updated every 10000/4 iterations !
DISPLAY_SCREEN = False
DYNAMIC_PLOT = False
VERBOSE = True
PLOTS_DIR = 'reward_plots/'
MODELS_DIR = 'saved_models/'
FILE_SUFFIX = 'lr' + str(LEARNING_RATE) + '_bs' + str(MINIBATCH_SIZE) + '_tu' + str(TARGET_UPDATE) + '_it' + \
              str(N_ITERATIONS) + '_g' + str(GAMMA) + '_ed' + str(EPS_DECAY_FRACTION_1)
SAVE_EVERY = 50000
SAVE_MODELS = [i for i in range(SAVE_EVERY, N_ITERATIONS - SAVE_EVERY + 1, SAVE_EVERY)]
N_NO_OP = 20# The agent performs between 1 and N_NO_OP NO_OP_ACTIONs at the start of each episode, point being to start in different situations
NO_OP_ACTION = 1# We choose the fire action so that the ball is launched, once in the game it does nothing anymore
# https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
