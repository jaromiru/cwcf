#============================================
#============ UNIFORM SETTINGS ==============
#============================================
import numpy as np
import importlib

class Config:
    def init(self, args):
        dataset = importlib.import_module("config_datasets." + args.dataset)
        self.dataset = dataset

        #================== GLOBAL SETTING
        self.BLANK_INIT = not args.load_progress
        self.USE_HPC    = args.use_hpc
        self.HARD_BUDGET = args.hard_budget
        self.PRETRAIN   = args.pretrain
        self.SEED       = args.seed
        self.FEATURE_FACTOR = args.target if args.target_type == 'lambda' else 0.
        self.TARGET_COST = args.target
        self.TARGET_TYPE = args.target_type
        self.MISSING_GRAD = args.missing_grad
        self.REWEIGHT   = args.reweight
        self.DEVICE     = args.device

        # ================== DATASET
        self.DATA_FILE = '../data/' + dataset.DATASET + '-train'
        self.DATA_VAL_FILE  = '../data/' + dataset.DATASET + '-val'
        self.DATA_TEST_FILE = '../data/' + dataset.DATASET + '-test'
        self.META_FILE = '../data/' + dataset.DATASET + '-meta'
        self.HPC_FILE = '../data/' + dataset.DATASET + '-hpc'

        self.CLASSES = dataset.CLASSES
        self.FEATURE_DIM = dataset.FEATURES
        self.ACTION_DIM = self.FEATURE_DIM + self.CLASSES + self.USE_HPC
        self.TERMINAL_ACTIONS = self.CLASSES + self.USE_HPC
        self.HPC_ACTION = self.CLASSES

        self.META_COSTS = 'cost'
        self.META_AVG   = 'avg'
        self.META_STD   = 'std'

        # ================== RL
        self.REWARD_CORRECT   =  0
        self.REWARD_INCORRECT = [-1] * self.CLASSES

        self.GAMMA  = 1.0
        self.LAMBDA = 1.0

        # ================== TRAINING
        self.AGENTS = 1000
        self.EPOCH_STEPS = 1

        self.MAX_TRAINING_EPOCHS = 100 * dataset.DIFFICULTY
        self.EVALUATE_STEPS   = 1 * dataset.DIFFICULTY        # compute r_avg over this many steps
        
        self.ACCURACY_CHECK   = 4       # check back 4 steps to improve accuracy
        self.LAGRANGIAN_CHECK_TRESHOLD = 1e-4    # lagrangian is stable
        self.LAGRANGIAN_CHECK_TIMES    = 3       # lagrangian is stable

        self.EPSILON_START  = 1.00
        self.EPSILON_END    = 0.10
        self.EPSILON_EPOCHS = 2 * dataset.DIFFICULTY    # epsilon will fall to EPSILON_END after EPSILON_EPOCHS
        self.EPSILON_UPDATE_EPOCHS = 10     # update epsilon every x epochs

        self.PI_EPSILON_START  = 0.50       # epsilon for the target policy
        self.PI_EPSILON_END    = 0.00
        self.PI_EPSILON_EPOCHS = 2 * dataset.DIFFICULTY

        # ================== LOG
        self.LOG_TRACKED_STATES = [np.zeros((2, self.FEATURE_DIM))]
        self.LOG_EPOCHS = 0.1 * dataset.DIFFICULTY           # states prediction will be logged every LOG_EPOCHS
        self.LOG_PERF_VAL_SIZE = -1

        # ================== NN
        self.BATCH_SIZE =   50000     # steps
        self.POOL_SIZE  =   40000     # episodes

        self.NN_FC_DENSITY = dataset.NN_SIZE
        self.NN_HIDDEN_LAYERS = 3

        self.OPT_LR = 5.0e-4
        self.OPT_L2 = 0.        
        self.OPT_ALPHA = 0.95
        self.OPT_MAX_NORM = 1.0

        self.OPT_LR_MIN = 5.0e-7
        self.OPT_LR_FACTOR =  0.5
        self.OPT_LR_STEPS = 10 * dataset.DIFFICULTY

        self.OPT_LAMBDA_LR = 1e-1
        self.OPT_LAMBDA_LR_MIN = 1e-4
        self.OPT_LAMBDA_EPOCHS = 0.1 * dataset.DIFFICULTY # update lambda every x epochs
        self.OPT_LAMBDA_GAMMA = 0.1 # momentum

        self.TARGET_RHO = 0.01

        # ================== PRETRAINING
        self.PRETRAIN_BATCH  = 1024
        self.PRETRAIN_EPOCHS = 10000
        self.PRETRAIN_CYCLES = 1
        self.PRETRAIN_ZERO_PROB = 3.0   # 66%
        self.PRETRAIN_LR = 1.0e-3
        self.PRETRAIN_LR_FACTOR = 0.1

        # ================== AUX
        self.SAVE_EPOCHS = 0.1 * dataset.DIFFICULTY
        self.MAX_MASK_CONST = 1.e12

        if hasattr(dataset, 'override'):
            for attr in vars(dataset.override):
                setattr(self, attr, getattr(dataset.override, attr))

    def print_short(self):
        short_keys = ['BLANK_INIT', 'USE_HPC', 'PRETRAIN', 'SEED', 'FEATURE_FACTOR']
        dataset_keys = ['DATASET', 'CLASSES', 'FEATURES', 'NN_SIZE', 'DIFFICULTY']

        print("Dataset configuration:")
        for key in dataset_keys:
            print("{}={}".format(key, vars(self.dataset)[key]), end=" ")

        print("\nGlobal configuration:")
        for key in short_keys:
            print("{}={}".format(key, vars(self)[key]), end=" ")

        # print("\nFull conf:")
        # print(vars(self))

        print("\n")


config = Config()
