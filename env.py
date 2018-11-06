import numpy as np

from config import config
#==============================

lin_array = np.arange(config.AGENTS)
empty_x = np.zeros(config.FEATURE_DIM, dtype=np.float32)
empty_n = np.zeros(config.FEATURE_DIM, dtype=np.bool)
no_class = -1

#==============================
class Environment:
    def __init__(self, data, hpc_p, costs):
        self.data_x = data.iloc[:, 0:-1].astype('float32').values
        self.data_n = np.isnan(self.data_x)
        self.data_x = np.nan_to_num(self.data_x)

        self.data_y = data.iloc[:,   -1].astype('int32').values

        self.data_len = len(data)

        self.hpc_p = hpc_p.values
        self.costs = costs.values

        self.mask = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.float32 )
        self.x    = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.float32 )
        self.y    = np.zeros( config.AGENTS, dtype=np.int64 )
        self.p    = np.zeros( config.AGENTS, dtype=np.int32 )
        self.n    = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.bool )

    def reset(self):
        for i in range(config.AGENTS):
            self._reset(i)

        s  = self._get_state(self.x, self.mask)
        na = self._get_actions(self.mask, self.n)
        return s, na

    def _reset(self, i):
        self.mask[i] = 0
        self.x[i], self.y[i], self.p[i], self.n[i] = self._generate_sample()

    def step(self, action):
        done = np.zeros(config.AGENTS, dtype=np.int8)
        corr = np.zeros(config.AGENTS, dtype=np.int8)
        hpc  = np.zeros(config.AGENTS, dtype=np.bool)
        hpc_fc = np.zeros(config.AGENTS, dtype=np.float32)
        eplen = np.sum(self.mask, axis=1) + 1

        mask_ = self.mask.copy()

        action_f = np.clip(action - config.TERMINAL_ACTIONS, 0, config.FEATURE_DIM)
        self.mask[lin_array, action_f] = 1
        r = -self.costs[action_f] * config.FEATURE_FACTOR

        for i in np.where(action < config.TERMINAL_ACTIONS)[0]:
            if config.USE_HPC and action[i] == config.HPC_ACTION:
                remaining_actions = (1 - self.n[i]) * (1 - mask_[i])
                r_feat = - np.sum( remaining_actions * self.costs ) * config.FEATURE_FACTOR              # total cost of remaining actions
                r_corr = config.REWARD_CORRECT if self.p[i] == self.y[i] else config.REWARD_INCORRECT

                hpc[i] = 1
                hpc_fc[i] = r_feat
                corr[i] = 1 if self.p[i] == self.y[i] else 0
                r[i] = r_feat + r_corr
            else:
                corr[i] = 1 if action[i] == self.y[i] else 0
                r[i] = config.REWARD_CORRECT if action[i] == self.y[i] else config.REWARD_INCORRECT

            done[i] = True
            self._reset(i)

        s_ = self._get_state(self.x, self.mask)
        info = {'corr':corr, 'hpc':hpc, 'hpc_fc':hpc_fc, 'eplen':eplen}
        na = self._get_actions(self.mask, self.n)

        return (s_, r, na, done, info)   # state, reward, unavailable actions, terminal flag, info dict

    def _generate_sample(self):
        idx = np.random.randint(0, self.data_len)

        x = self.data_x[idx]        # sample features
        y = self.data_y[idx]        # class
        p = self.hpc_p[idx]         # HPC prediction
        n = self.data_n[idx]        # nan features

        return (x, y, p, n)

    @staticmethod
    def _get_state(x, m):
        x_ = (x * m).reshape(-1, 1, config.FEATURE_DIM)
        m_ = m.reshape(-1, 1, config.FEATURE_DIM)

        s = np.concatenate( (x_, m_), axis=1 ).astype(np.float32)
        return s

    @staticmethod
    def _get_actions(m, n):
        a = np.zeros((config.AGENTS, config.ACTION_DIM), dtype=np.float32)
        a[:, config.TERMINAL_ACTIONS:] = m + n

        return a

    @staticmethod
    def _random_mask(size, zero_prob):
        mask_p = np.random.rand() ** zero_prob  # ratio of ones
        mask_rand = np.random.rand(size, config.FEATURE_DIM)

        mask = np.zeros((size, config.FEATURE_DIM), dtype='float32')
        mask[ mask_rand < mask_p ] = 1

        return mask

    def _get_random_batch(self, size, zero_prob):
        idx = np.random.randint(len(self.data_x), size=size)
        x = self.data_x[idx]
        y = self.data_y[idx]
        p = self.hpc_p[idx]
        n = self.data_n[idx]

        m = Environment._random_mask(size, zero_prob) * ~n  # can take only available features
        s = Environment._get_state(x, m)

        a = ~np.logical_or(m, n)                        # available actions
        c = np.sum(a * self.costs * config.FEATURE_FACTOR, axis=1)    # cost of remaining actions

        return (s, x, y, p, c)

#==============================
class SeqEnvironment(Environment):
    def reset(self):
        self.idx = 0
        return super().reset()

    def _generate_sample(self):
        if self.idx >= self.data_len:
            return (empty_x, no_class, no_class, empty_n)
        else:
            x = self.data_x[self.idx]
            y = self.data_y[self.idx]
            p = self.hpc_p[self.idx]
            n = self.data_n[self.idx]

            self.idx += 1

            return (x, y, p, n)

    def step(self, action):
        terminated = self.y == no_class

        s_, r, a, done, info = super().step(action)

        # flag terminated
        done[terminated] = -1
        r[terminated] = 0
        info['corr'][terminated] = 0
        info['hpc'][terminated] = 0
        info['hpc_fc'][terminated] = 0

        return (s_, r, a, done, info)
