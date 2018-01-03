BLANK_INIT = True

#================== DATASET
DATA_FILE = '../data/mb-train'
DATA_VAL_FILE = '../data/mb-val'
META_FILE = '../data/mb-meta'

CLASSES = 2
FEATURE_DIM = 50
STATE_DIM = FEATURE_DIM * 2
ACTION_DIM = FEATURE_DIM + CLASSES

COLUMN_LABEL = '_label'
COLUMN_DROP  = ['_count']

META_COSTS = 'cost'
META_AVG   = 'avg'
META_STD   = 'std'
 
#================== RL
FEATURE_FACTOR   =   0.001
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1

#================== TRAINING
AGENTS = 1000

TRAINING_EPOCHS = 10000

EPOCH_STEPS = 1

EPSILON_START  = 0.50
EPSILON_END    = 0.05
EPSILON_EPOCHS = 2000	 	# epsilon will fall to EPSILON_END after EPSILON_EPOCHS
EPSILON_UPDATE_EPOCHS = 10  # update epsilon every x epochs

#================== LOG
from log_states.log_mb import TRACKED_STATES
LOG_TRACKED_STATES = TRACKED_STATES

LOG_EPOCHS = 100  			# states prediction will be logged every LOG_EPOCHS

LOG_PERF_EPOCHS = 100
LOG_PERF_VAL_SIZE = 1000

#================== NN
BATCH_SIZE =    100000
POOL_SIZE  =   2000000

NN_FC_DENSITY = 128
NN_HIDDEN_LAYERS = 3

OPT_LR = 1.0e-4
OPT_ALPHA = 0.95
OPT_MAX_NORM = 1.0

# LR scheduling => lower LR by LR_SC_FACTOR every LR_SC_EPOCHS epochs
LR_SC_FACTOR =  0.1
LR_SC_EPOCHS = 5000
LR_SC_MIN = 1.0e-7

TARGET_RHO = 0.01

#================== AUX
SAVE_EPOCHS = 1000
MAX_MASK_CONST = 1.e6

SEED = 112233
