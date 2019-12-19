RANDOM_SEED = 1
N_ENTRIES = 4
N_ACTIONS = 2
#FEATURES_SIZES = [N_ENTRIES, 12, 24, 16, 8, N_ACTIONS]
FEATURES_SIZES = [N_ENTRIES, 8, 4, N_ACTIONS]
MINIBATCH_SIZE = 512
N_ITERATIONS = 400000
RM_CAPACITY = N_ITERATIONS
INITIAL_EPSILON = 1.
MINIMAL_EPSILON = .05
FACTOR_TO_MIN_EPS = 2.5
EPSILON_ANNEALING_STEP = (INITIAL_EPSILON - MINIMAL_EPSILON) / ((1. / FACTOR_TO_MIN_EPS) * N_ITERATIONS)
GAMMA = .999# was .999 since the start, .98 too low, maybe .99 too, .995
LEARNING_RATE = .0001
TARGET_UPDATE = 10000# 4000 too high, 500 too small, 2000 quite OK, 2500 to test again
DISPLAY_SCREEN = True
DYNAMIC_PLOT = True
VERBOSE = True
PLOTS_DIR = 'reward_plots/'
MODELS_DIR = 'saved_models/'
FILE_SUFFIX = 'lightmodel_clipping_bigrm_lr' + str(LEARNING_RATE) + '_bs' + str(MINIBATCH_SIZE) + \
              '_tu' + str(TARGET_UPDATE) + '_it' + str(N_ITERATIONS) + '_g' + str(GAMMA)
SAVE_MODELS = []#N_ITERATIONS // 2, (3 * N_ITERATIONS) // 4, (7 * N_ITERATIONS) // 8]
USE_MODEL = ''#'saved_models/policy_net_lightmodel_clipping_bigrm_lr0.0001_bs512_tu10000_it400000_g0.999_c350000.pt'
