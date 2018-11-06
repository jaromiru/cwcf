DATASET = 'cifar'

CLASSES  = 10
FEATURES = 400

NN_SIZE    = 512
DIFFICULTY = 10000

class Override():
    def __init__(self):
        self.POOL_SIZE  = 10000

        self.PRETRAIN_LR = 2.0e-5
        self.OPT_LR = 1.0e-5


override = Override()
