import random

class Pool():
    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.idx  = 0

        self.sum_len = 0

        self.total = 0

    def put(self, x):
        if(self.total >= self.size):
            old_x = self.data[self.idx]
            self.sum_len -= len(old_x[0])

        self.sum_len += len(x[0])

        self.data[self.idx] = x
        self.idx = (self.idx + 1) % self.size
        self.total += 1

    ''' Sample a batch of #size episodes. '''
    def sample(self, size):
        return random.choices(self.data, k=size)

    ''' Samples a batch of episodes. Size of total steps is close to #size.'''
    def sample_steps(self, size):
        avg_len = self.sum_len / self.size
        eps_to_fetch = int(size / avg_len)

        return random.choices(self.data, k=eps_to_fetch)
