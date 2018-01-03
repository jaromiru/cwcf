BLANK_INIT = True

#================== DATASET
DATA_FILE = '../data/cifar-train'
DATA_VAL_FILE = '../data/cifar-val'
META_FILE = '../data/cifar-meta'

CLASSES = 2
FEATURE_DIM = 400
STATE_DIM = FEATURE_DIM * 2
ACTION_DIM = FEATURE_DIM + CLASSES

COLUMN_LABEL = '_label'
COLUMN_DROP  = []

META_COSTS = 'cost'
META_AVG   = 'avg'
META_STD   = 'std'

#================== RL
FEATURE_FACTOR   =   0.0001
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1		

#================== TRAINING
AGENTS = 1000

TRAINING_EPOCHS = 2000000

EPOCH_STEPS = 1

EPSILON_START  = 1.00
EPSILON_END    = 0.10
EPSILON_EPOCHS = 100000	 	 # epsilon will fall to EPSILON_END after EPSILON_EPOCHS
EPSILON_UPDATE_EPOCHS = 100  # update epsilon every x epochs

#================== LOG
from log_states.log_cifar import TRACKED_STATES
LOG_TRACKED_STATES = TRACKED_STATES

LOG_EPOCHS = 100  			# states prediction will be logged every LOG_EPOCHS

LOG_PERF_EPOCHS = 1000
LOG_PERF_VAL_SIZE = 1000

#================== NN
BATCH_SIZE =   100000
POOL_SIZE  =  1000000

NN_FC_DENSITY = 512
NN_HIDDEN_LAYERS = 3

OPT_LR = 1.0e-6
OPT_ALPHA = 0.95
OPT_MAX_NORM = 1.0

# LR scheduling => lower LR by LR_SC_FACTOR every LR_SC_EPOCHS epochs
LR_SC_FACTOR =   0.9
LR_SC_EPOCHS = 10000
LR_SC_MIN = 3.0e-7

TARGET_RHO = 0.01

#================== AUX
SAVE_EPOCHS = 1000
MAX_MASK_CONST = 1.e6

SEED = 112233
