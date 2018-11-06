import numpy as np
from config import config

all_agents = np.arange(config.AGENTS)

class Agent():
    def __init__(self, env, pool, brain):
        self.env  = env
        self.pool = pool
        self.brain = brain

        self.epsilon = config.EPSILON_START

        self.idx = np.zeros(config.AGENTS, dtype=np.int32)
        self.S   = np.zeros((config.AGENTS, config.FEATURE_DIM+1, 2, config.FEATURE_DIM), dtype=np.float32)
        self.A   = np.zeros((config.AGENTS, config.FEATURE_DIM+1), dtype=np.int64)
        self.R   = np.zeros((config.AGENTS, config.FEATURE_DIM+1), dtype=np.float32)
        self.U   = np.zeros((config.AGENTS, config.FEATURE_DIM+1), dtype=np.float32)
        self.NA  = np.zeros((config.AGENTS, config.FEATURE_DIM+1, config.ACTION_DIM), dtype=np.bool)

        s, na = self.env.reset()
        self.S[all_agents, self.idx] = s
        self.NA[all_agents, self.idx] = na

    def act(self, s, na):
        q = self.brain.predict_np(s)
        p = q - config.MAX_MASK_CONST * na 	# select an action not considering those already performed
        a = np.argmax(p, axis=1)

        rand_agents = np.random.rand(config.AGENTS) < self.epsilon
        rand_number = np.random.rand(config.AGENTS)					# rand() call is expensive, better to do it at once

        possible_actions_count = config.ACTION_DIM - np.sum(na, axis=1)
        u = (1 - self.epsilon) + (self.epsilon / possible_actions_count)

        for i in range(config.AGENTS):
            if rand_agents[i]:  # random action
                possible_actions = np.where( na[i] == False )[0]    # select a random action, don't repeat an action

                w = int(rand_number[i] * possible_actions_count[i])
                a_ = possible_actions[w]

                if a[i] == a_:
                    u[i] = (1 - self.epsilon) + (self.epsilon / possible_actions_count[i])  # randomly selected the maximizing action

                else:
                    a[i] = a_
                    u[i] = self.epsilon / possible_actions_count[i]  # probability of taking a random action

        return a, u

    def step(self):
        s = self.S[all_agents, self.idx]
        na = self.NA[all_agents, self.idx]

        a, u = self.act(s, na)
        s_, r, na_, done, info = self.env.step(a)

        self.A[all_agents, self.idx] = a
        self.R[all_agents, self.idx] = r
        self.U[all_agents, self.idx] = u

        for i in np.where(done)[0]:     # truncate & store the finished episode i
            idx = self.idx[i]+1

            _s = self.S[i, :idx].copy()
            _a = self.A[i, :idx].copy()
            _r = self.R[i, :idx].copy()
            _u = self.U[i, :idx].copy()
            _na = self.NA[i, :idx].copy()

            # extract the true state
            _x = np.broadcast_to(self.env.x[i].copy(), (idx, config.FEATURE_DIM))
            _y = np.repeat(self.env.y[i], idx)

            self.pool.put( (_s, _a, _r, _u, _na, _x, _y) )

        self.idx = (done == 0) * (self.idx + 1)     # advance idx by 1 and reset to 0 for finished episodes

        self.NA[all_agents, self.idx] = na_     # unavailable actions
        self.S[all_agents,  self.idx] = s_

        return s, a, r, s_, done, info

    def update_epsilon(self, epoch):
        if epoch >= config.EPSILON_EPOCHS:
            self.epsilon = config.EPSILON_END
        else:
            self.epsilon = config.EPSILON_START + epoch * (config.EPSILON_END - config.EPSILON_START) / config.EPSILON_EPOCHS

class PerfAgent(Agent):
    def __init__(self, env, brain):
        self.env  = env
        self.brain = brain

        self.idx = np.zeros(config.AGENTS, dtype=np.int32)
        self.S   = np.zeros((config.AGENTS, config.FEATURE_DIM+1, 2, config.FEATURE_DIM), dtype=np.float32)
        self.NA  = np.zeros((config.AGENTS, config.FEATURE_DIM+1, config.ACTION_DIM), dtype=np.bool)

        s, na = self.env.reset()
        self.S[all_agents, self.idx] = s
        self.NA[all_agents, self.idx] = na

    def act(self, s, na):
        q = self.brain.predict_np(s)
        p = q - config.MAX_MASK_CONST * na 	# select an action not considering those already performed
        a = np.argmax(p, axis=1)

        return a, 1.0

    def step(self):
        s = self.S[all_agents, self.idx]
        na = self.NA[all_agents, self.idx]

        a, u = self.act(s, na)
        s_, r, na_, done, info = self.env.step(a)

        self.idx = (done == 0) * (self.idx + 1)     # advance idx by 1 and reset to 0 for finished episodes

        self.NA[all_agents, self.idx] = na_         # unavailable actions
        self.S[all_agents, self.idx] = s_

        return s, a, r, s_, done, info
