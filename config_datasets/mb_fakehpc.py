DATASET = 'mb'

CLASSES  = 2
FEATURES = 50

NN_SIZE    = 128
DIFFICULTY = 1000

class Override():
    def __init__(self):
        self.HPC_FILE = '../data/' + DATASET + '-hpc-fake'


override = Override()