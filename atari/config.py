N_ACTIONS = 3
MINIBATCH_SIZE = 32# 128
N_ITERATIONS = 300000
RM_CAPACITY = N_ITERATIONS
INITIAL_EPSILON = 1.
MINIMAL_EPSILON = .1
EPS_DECAY_FRACTION = .5
EPSILON_ANNEALING_STEP = (INITIAL_EPSILON - MINIMAL_EPSILON) / (EPS_DECAY_FRACTION * N_ITERATIONS)
GAMMA = .99
LEARNING_RATE = .00025# .001
TARGET_UPDATE = 10000
K_SKIP_FRAMES = 4
DISPLAY_SCREEN = False
DYNAMIC_PLOT = False
VERBOSE = True
PLOTS_DIR = 'reward_plots/'
MODELS_DIR = 'saved_models/'
FILE_SUFFIX = 'lr' + str(LEARNING_RATE) + '_bs' + str(MINIBATCH_SIZE) + \
              '_tu' + str(TARGET_UPDATE) + '_it' + str(N_ITERATIONS)
SAVE_MODELS = [50000, 100000, 150000, 200000]
N_NO_OP = 30
NO_OP_ACTION = 1
