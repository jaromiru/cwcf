DATASET = 'forest-2'

CLASSES  = 2
FEATURES = 54

NN_SIZE    = 256
DIFFICULTY = 10000

class Override():
    def __init__(self):
        self.HPC_FILE = '../data/' + DATASET + '-hpc-fake'

override = Override()