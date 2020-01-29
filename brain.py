import numpy as np
import torch

from config import config

from net import Net
import utils

#==============================
class Brain:
    def __init__(self, pool):
        self.pool = pool

        self.model  = Net()
        self.model_ = Net()

        self.epsilon = config.PI_EPSILON_START
        self.lr = config.OPT_LR

        print("Network architecture:\n" + str(self.model))

    def _load(self, file='model'):
        if config.DEVICE == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(config.DEVICE)

        self.model.load_state_dict( torch.load(file, map_location=device) )
        self.model_.load_state_dict( torch.load(file + "_", map_location=device) )

    def _save(self, file='model'):
        torch.save(self.model.state_dict(), file)
        torch.save(self.model_.state_dict(),file + "_")

    def predict_pt(self, s, target):
        if target:
            return self.model_(s)
        else:
            return self.model(s)

    def predict_np(self, s, target=False):
        s = torch.from_numpy(s)
        q = self.predict_pt(s, target)

        q = q.detach().cpu().numpy()

        return q

    def train(self):
        batch = self.pool.sample_steps(config.BATCH_SIZE)

        _s, _a, _r, _u, _na, _w, _x, _y = zip(*batch)

        _s = np.vstack(_s)
        _a = np.concatenate(_a)
        _r = np.concatenate(_r)
        _u = np.concatenate(_u)
        _na = np.vstack(_na).astype(np.float32)
        _w = np.concatenate(_w)

        # rescale reward
        _r[_a >= config.TERMINAL_ACTIONS] *= config.FEATURE_FACTOR
        _a = _a.reshape(-1, 1)
        # _x = np.vstack(_x)
        # _y = np.concatenate(_y)

        # ratio = np.mean(_y == 0)
        # print("CLS 0: {:.2f}".format(ratio))

        # push to GPU
        s = torch.from_numpy(_s).to(self.model.device)
        a = torch.from_numpy(_a).to(self.model.device)
        r = torch.from_numpy(_r).to(self.model.device)
        u = torch.from_numpy(_u).to(self.model.device)
        na = torch.from_numpy(_na).to(self.model.device)   # actions not available
        w = torch.from_numpy(_w).to(self.model.device)

        # x = torch.from_numpy(_x).to(self.model.device)
        # y = torch.from_numpy(_y).to(self.model.device)

        batch_len = _s.shape[0]

        # extract the mask
        # m = torch.FloatTensor(batch_len, ACTION_DIM).zero_().to(self.model.device)
        # m[:, TERMINAL_ACTIONS:] = s[:, 1]
        # either use 'm' or 'na' for to use unavailable features information or not

        # compute
        q_orig = self.predict_pt(s, target=False)
        q_current = q_orig.detach() - (config.MAX_MASK_CONST * na) 			      		        # unavailable actions do not influence the max

        q_target = self.predict_pt(s, target=True)
        q_target = q_target.detach()

        _, a_max = q_current.max(dim=1, keepdim=True)
        a_count = config.ACTION_DIM - na.sum(dim=1, keepdim=True)

        p_matrix = (1.0 - na) * (self.epsilon / a_count)						          		# matrix of p(a|s)
        p_matrix.scatter_(1, a_max, (1 - self.epsilon) + (self.epsilon / a_count))				# p(a* | s)

        p = p_matrix.gather(1, a).view(-1)														# p(a|s)

        q_estm = torch.sum(p_matrix * q_target, dim=1, keepdim=False)   					 	# E_pi[ Q(s, *) ]
        q_perf = q_target.gather(1, a).view(-1)												    # performed value

        c = config.LAMBDA * (p / u).clamp_(max=1.0)

        # compute q in a parallel way
        terminal = torch.ones(batch_len, dtype=torch.float32)   # terminal flag (0 when terminal)
        idxs     = torch.zeros(len(batch), dtype=torch.int64)

        max_ep_len = 0
        start = 0

        for idx in range(len(batch)):
            ep_len = len(batch[idx][0])
            end = start + ep_len - 1
            start = end + 1

            terminal[end] = 0.
            idxs[idx] = end
            max_ep_len = max(max_ep_len, ep_len)

        terminal = terminal.to(self.model.device)
        idxs = idxs.to(self.model.device)

        q = torch.zeros(batch_len, dtype=torch.float32, device=self.model.device)
        q[idxs] = r[idxs]
        for i in range(1, max_ep_len):
            idxs -= 1
            q[idxs] = r[idxs] + terminal[idxs] * config.GAMMA * q_estm[idxs+1] + terminal[idxs] * config.GAMMA * c[idxs+1] * (q[idxs+1] - q_perf[idxs+1])

        q.clamp_(max=0)		# bind the values to theoretical q function range

        # compute targets for A_c actions
        # q_c = np.full((batch_len, CLASSES), REWARD_INCORRECT, dtype=np.float32)
        # q_c[np.arange(batch_len), _y] = REWARD_CORRECT
        # q_c = torch.from_numpy(q_c).to(self.model.device)

        # train
        self.model.train_pred(q_orig, a, q, w)
        self.model_.copy_weights(self.model)

    # TODO rebalanced pretraining?
    def _get_batch(self, env, size):
        s, x, y, p, c, w = env._get_random_batch(size, config.PRETRAIN_ZERO_PROB)

        # print("Class distribution:")
        # for y_ in np.unique(y):
        #     print(np.mean(y==y_))

        # TODO
        q_c = np.full((size, config.TERMINAL_ACTIONS), config.REWARD_INCORRECT, dtype=np.float32)
        q_c[np.arange(size), y] = config.REWARD_CORRECT

        if config.USE_HPC:
            q_c[p == y, config.HPC_ACTION] = config.REWARD_CORRECT
            q_c[:, config.HPC_ACTION] -= c

        # make it cuda
        s   = torch.from_numpy(s).to(self.model.device)
        q_c = torch.from_numpy(q_c).to(self.model.device)

        if type(w) is not float:
            w = torch.from_numpy(w).to(self.model.device)

        # x   = torch.from_numpy(x).to(self.model.device)
        # y   = torch.from_numpy(y.astype(np.int64)).to(self.model.device) # loss_cross expects LongTensor

        return s, q_c, w

    def pretrain(self, env):
        test_s, test_q_c, test_w = self._get_batch(env, config.PRETRAIN_BATCH)

        last_loss = 9999.
        for i in range(config.PRETRAIN_EPOCHS):
            utils.print_progress(i, config.PRETRAIN_EPOCHS, step=100)

            # print loss
            if utils.is_time(i, 100):
                lr = config.PRETRAIN_LR / (1 + i / 100)
                self.model.set_lr(lr)

                q = self.model(test_s)
                loss_c = self.model.get_loss_c(q, test_q_c, test_w)
                print("\nLoss: {:.4e} LR: {:.2e}".format(loss_c.data.item(), lr))

                if last_loss <= loss_c:     # stop early
                    break

                last_loss = loss_c

            s, q_c, w = self._get_batch(env, config.PRETRAIN_BATCH)
            self.model.train_c(s, q_c, w)

        self.model_.copy_weights(self.model, rho=1.0)

    # def update_lr(self, epoch):
    #     lr = config.OPT_LR * (config.OPT_LR_FACTOR ** (epoch // config.LR_SC_EPOCHS))
    #     lr = max(lr, config.LR_SC_MIN)

    #     self.model.set_lr(lr)
    #     print("Setting LR: {:.2e}".format(lr))

    def set_lr(self, lr):
        self.lr = max(lr, config.OPT_LR_MIN)
        self.model.set_lr(self.lr)
        print("Setting LR: {:.2e}".format(self.lr))        

    def lower_lr(self):
        self.lr = max(self.lr * config.OPT_LR_FACTOR, config.OPT_LR_MIN)
        self.model.set_lr(self.lr)
        print("Setting LR: {:.2e}".format(self.lr))

    def update_epsilon(self, epoch):
        if epoch >= config.PI_EPSILON_EPOCHS:
            self.epsilon = config.PI_EPSILON_END
        else:
            self.epsilon = config.PI_EPSILON_START + epoch * (config.PI_EPSILON_END - config.PI_EPSILON_START) / config.PI_EPSILON_EPOCHS
