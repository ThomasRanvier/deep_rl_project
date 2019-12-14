RM_CAPACITY = 10000
N_ENTRIES = 4
N_ACTIONS = 2
FEATURES_SIZES = [N_ENTRIES, 12, 24, 16, 8, N_ACTIONS]
MINIBATCH_SIZE = 64
INITIAL_EPSILON = 1.
EPSILON_ANNEALING_STEP = .01
MINIMAL_EPSILON = .1
GAMMA = .999
LEARNING_RATE = .001
N_EPISODES = 400
TARGET_UPDATE = 1