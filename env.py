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

        self.data_y = data.iloc[:, -1].astype('int32').values
        self.data_len = len(data)

        if config.REWEIGHT != 'no':
            self.data_w = np.zeros(self.data_len, dtype=np.float32)
            for c in range(config.CLASSES):
                c_idx = self.data_y == c
                c_w = 1 / np.sum(c_idx)
                self.data_w[c_idx] = c_w

            self.data_w = self.data_w / sum(self.data_w)

        self.hpc_p = hpc_p.values
        self.costs = costs.values

        self.mask = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.float32 )   # already selected actions
        self.x    = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.float32 )   # sample
        self.y    = np.zeros( config.AGENTS, dtype=np.int64 )                           # class
        self.p    = np.zeros( config.AGENTS, dtype=np.int32 )                           # hpc predictions
        self.n    = np.zeros( (config.AGENTS, config.FEATURE_DIM), dtype=np.bool )      # missing values
        self.b    = np.zeros( config.AGENTS, dtype=np.float32 )                         # remaining budget
        self.w    = np.zeros( config.AGENTS, dtype=np.float32 )                         # reweighting correction

    def reset(self):    
        for i in range(config.AGENTS):
            self._reset(i)

        actions_overbudget = self.costs > self.b.reshape(-1, 1)

        s  = self._get_state(self.x, self.mask)
        na, na_grad = self._get_actions(self.mask, self.n, actions_overbudget)

        return s, na, na_grad

    def _reset(self, i):
        self.mask[i] = 0
        self.b[i] = config.HARD_BUDGET

        self.x[i], self.y[i], self.p[i], self.n[i], self.w[i] = self._generate_sample()

    def step(self, action):
        done = np.zeros(config.AGENTS, dtype=np.int8)
        corr = np.zeros(config.AGENTS, dtype=np.int8)
        hpc  = np.zeros(config.AGENTS, dtype=np.bool)
        hpc_fc = np.zeros(config.AGENTS, dtype=np.float32)
        fc = np.zeros(config.AGENTS, dtype=np.float32)
        eplen = np.sum(self.mask, axis=1) + 1
        pred_y = np.full(config.AGENTS, None)
        true_y = np.full(config.AGENTS, None)

        w = self.w.copy()   # save old correction
        mask_ = self.mask.copy()

        action_f = np.clip(action - config.TERMINAL_ACTIONS, 0, config.FEATURE_DIM)

        # catch errors: action is either classification or it hasn't been selected before
        assert np.all((action < config.TERMINAL_ACTIONS) + (self.mask[lin_array, action_f] == 0))

        self.mask[lin_array, action_f] = 1
        self.b -= self.costs[action_f]

        r = -self.costs[action_f] # * config.FEATURE_FACTOR    
        fc = self.costs[action_f]   

        for i in np.where(action < config.TERMINAL_ACTIONS)[0]:
            if config.USE_HPC and action[i] == config.HPC_ACTION:
                a = ~np.logical_or(mask_[i], self.n[i])                       # available actions
                c = -np.sum(a * self.costs) * config.FEATURE_FACTOR           # cost of remaining actions

                r_corr = config.REWARD_CORRECT if self.p[i] == self.y[i] else config.REWARD_INCORRECT[self.y[i]]

                hpc[i] = 1
                hpc_fc[i] = c
                pred_y[i] = self.p[i]
                true_y[i] = self.y[i]

                corr[i] = 1 if self.p[i] == self.y[i] else 0
                r[i] = c + r_corr
                fc[i] = 0

            else:
                pred_y[i] = action[i]
                true_y[i] = self.y[i]

                corr[i] = 1 if action[i] == self.y[i] else 0
                r[i] = config.REWARD_CORRECT if action[i] == self.y[i] else config.REWARD_INCORRECT[self.y[i]]
                fc[i] = 0

            done[i] = True
            self._reset(i)

        s_ = self._get_state(self.x, self.mask)
        info = {'corr':corr, 'hpc':hpc, 'fc':fc, 'hpc_fc':hpc_fc, 'eplen':eplen, 'pred_y':pred_y, 'true_y':true_y}

        actions_overbudget = self.costs > self.b.reshape(-1, 1)
        na, na_grad = self._get_actions(self.mask, self.n, actions_overbudget)

        # print("Costs:", self.costs)
        # print("Budget:", self.b)

        # print("Mask:", self.mask)
        # print("Over:", actions_overbudget)
        # print()
        # input()

        return (s_, r, na, na_grad, w, done, info)   # state, reward, unavailable actions, terminal flag, info dict

    def _generate_sample(self):
        if config.REWEIGHT == 'no':
            idx = np.random.randint(0, self.data_len)
            w = 1.
        else:           
            idx = np.random.choice(self.data_len, p=self.data_w) # TODO: this will be slow ; better prepare classes separately and choose from the bag
            if config.REWEIGHT == 'data':
                w = 1.
            else:   # data_gradient
                w = (1 / self.data_len) / self.data_w[idx] # bias correction = p_orig / p_rebalanced

        x = self.data_x[idx]        # sample features
        y = self.data_y[idx]        # class
        p = self.hpc_p[idx]         # HPC prediction
        n = self.data_n[idx]        # nan features

        return (x, y, p, n, w)

    @staticmethod
    def _get_state(x, m):
        x_ = (x * m).reshape(-1, 1, config.FEATURE_DIM)
        m_ = m.reshape(-1, 1, config.FEATURE_DIM)

        s = np.concatenate( (x_, m_), axis=1 ).astype(np.float32)
        return s


    ''' Returns a binary array indicating which actions are unavailable (m = already selected actions, n = missing values, b = actions exceeding hard budget). '''
    @staticmethod
    def _get_actions(m, n, b):
        a = np.zeros((config.AGENTS, config.ACTION_DIM), dtype=np.float32)

        if config.MISSING_GRAD == 'sparse':
            a[:, config.TERMINAL_ACTIONS:] = m + n + b
        else:                     # 'full'
            a[:, config.TERMINAL_ACTIONS:] = m + b

        na      = a 
        na_grad = a        

        return na, na_grad

    @staticmethod
    def _random_mask(size, zero_prob):
        mask_p = np.random.rand() ** zero_prob  # ratio of ones
        mask_rand = np.random.rand(size, config.FEATURE_DIM)

        mask = np.zeros((size, config.FEATURE_DIM), dtype='float32')
        mask[ mask_rand < mask_p ] = 1

        return mask

    def _get_random_batch(self, size, zero_prob):
        if config.REWEIGHT == 'no':      
            idx = np.random.randint(len(self.data_x), size=size)
            w = 1.
        else:
            idx = np.random.choice(self.data_len, size=size, p=self.data_w)
            if config.REWEIGHT == 'data':
                w = 1.
            else:   # data_gradient
                w = (1 / self.data_len) / self.data_w[idx] # bias correction = p_orig / p_rebalanced


        x = self.data_x[idx]
        y = self.data_y[idx]
        p = self.hpc_p[idx]
        n = self.data_n[idx]

        m = Environment._random_mask(size, zero_prob) * ~n            # can take only available features
        s = Environment._get_state(x, m)

        a = ~np.logical_or(m, n)                                      # available actions
        c = np.sum(a * self.costs * config.FEATURE_FACTOR, axis=1)    # cost of remaining actions

        return (s, x, y, p, c, w)

#==============================
class SeqEnvironment(Environment):
    def reset(self):
        self.idx = 0
        s, na, na_grad = super().reset()   

        return s, na

    def _generate_sample(self):
        if self.idx >= self.data_len:
            return (empty_x, no_class, no_class, empty_n, 0.0)
        else:
            x = self.data_x[self.idx]
            y = self.data_y[self.idx]
            p = self.hpc_p[self.idx]
            n = self.data_n[self.idx]

            if config.REWEIGHT == 'data':
                w = self.data_w[self.idx] / (1 / self.data_len)  # bias correction = p_rebalanced / p_orig <-- for statistics, regard the dataset as balanced
            else:
                w = 1.

            self.idx += 1

            return (x, y, p, n, w)

    def step(self, action):
        terminated = self.y == no_class

        s_, r, na, na_grad, w, done, info = super().step(action)

        # flag terminated
        done[terminated] = -1
        r[terminated] = 0
        info['corr'][terminated] = 0
        info['hpc'][terminated] = 0
        info['hpc_fc'][terminated] = 0
        info['fc'][terminated] = 0
        info['true_y'][terminated] = None
        info['pred_y'][terminated] = None

        return (s_, r, na, w, done, info)
