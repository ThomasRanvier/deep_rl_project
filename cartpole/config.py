RANDOM_SEED = 1
N_ENTRIES = 4
N_ACTIONS = 2
#FEATURES_SIZES = [N_ENTRIES, 12, 24, 16, 8, N_ACTIONS]
FEATURES_SIZES = [N_ENTRIES, 8, 4, N_ACTIONS]
MINIBATCH_SIZE = 512#512
N_ITERATIONS = 800000#400000
RM_CAPACITY = N_ITERATIONS
INITIAL_EPSILON = 1.#1.
MINIMAL_EPSILON = .05
EPS_DECAY_FRACTION = .25# .4
EPSILON_ANNEALING_STEP = (INITIAL_EPSILON - MINIMAL_EPSILON) / (EPS_DECAY_FRACTION * N_ITERATIONS)
GAMMA = .999# was .999 since the start, .98 too low, maybe .99 too, .995
LEARNING_RATE = .0001# .0001 OK, testing with .00005
TARGET_UPDATE = 10000# 10000 for big N_ITERATIONS, 2000 otherwise
DISPLAY_SCREEN = False
DYNAMIC_PLOT = False
VERBOSE = True
PLOTS_DIR = 'reward_plots/'
MODELS_DIR = 'saved_models/'
FILE_SUFFIX = 'hardscalingr_lr' + str(LEARNING_RATE) + '_bs' + str(MINIBATCH_SIZE) + \
              '_tu' + str(TARGET_UPDATE) + '_it' + str(N_ITERATIONS) + '_g' + str(GAMMA) + '_ed' + str(EPS_DECAY_FRACTION)
SAVE_MODELS = [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000]#N_ITERATIONS // 2, (3 * N_ITERATIONS) // 4, (7 * N_ITERATIONS) // 8]
