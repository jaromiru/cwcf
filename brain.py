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
        self.model.load_state_dict( torch.load(file) )
        self.model_.load_state_dict( torch.load(file + "_") )

    def _save(self, file='model'):
        torch.save(self.model.state_dict(), file)
        torch.save(self.model_.state_dict(),file + "_")

    def predict_pt(self, s, target):
        if target:
            return self.model_(s)
        else:
            return self.model(s)

    def predict_np(self, s, target=False):
        s = torch.from_numpy(s).cuda()
        q = self.predict_pt(s, target)

        q = q.detach().cpu().numpy()

        return q

    def train(self):
        batch = self.pool.sample_steps(config.BATCH_SIZE)

        _s, _a, _r, _u, _na, _x, _y = zip(*batch)

        _s = np.vstack(_s)
        _a = np.concatenate(_a).reshape(-1, 1)
        _r = np.concatenate(_r)
        _u = np.concatenate(_u)
        _na = np.vstack(_na).astype(np.float32)

        # push to GPU
        s = torch.from_numpy(_s).cuda()
        a = torch.from_numpy(_a).cuda()
        r = torch.from_numpy(_r).cuda()
        u = torch.from_numpy(_u).cuda()
        na = torch.from_numpy(_na).cuda()

        batch_len = _s.shape[0]

        m = na

        # compute
        q_orig = self.predict_pt(s, target=False)
        q_current = q_orig.detach() - (config.MAX_MASK_CONST * m) 					# unavailable actions do not influence the max

        q_target = self.predict_pt(s, target=True)
        q_target = q_target.detach()

        _, a_max = q_current.max(dim=1, keepdim=True)
        a_count = config.ACTION_DIM - m.sum(dim=1, keepdim=True)

        p_matrix = (1.0 - m) * (self.epsilon / a_count)											# matrix of p(a|s)
        p_matrix.scatter_(1, a_max, (1 - self.epsilon) + (self.epsilon / a_count))				# p(a* | s)

        p = p_matrix.gather(1, a).view(-1)														# p(a|s)

        q_estm = torch.sum(p_matrix * q_target, dim=1, keepdim=False)   					 	# E_pi[ Q(s, *) ]
        q_perf = q_target.gather(1, a).view(-1)												    # performed value

        c = config.LAMBDA * (p / u).clamp_(max=1.0)

        # compute q in a parallel way
        q = torch.cuda.FloatTensor(batch_len)

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

        terminal = terminal.cuda()
        idxs = idxs.cuda()

        q[idxs] = r[idxs]
        for i in range(1, max_ep_len):
            idxs -= 1
            q[idxs] = r[idxs] + terminal[idxs] * config.GAMMA * q_estm[idxs+1] + terminal[idxs] * config.GAMMA * c[idxs+1] * (q[idxs+1] - q_perf[idxs+1])

        q.clamp_(max=0)		# bind the values to theoretical q function range

        # train
        self.model.train_pred(q_orig, a, q)
        self.model_.copy_weights(self.model)

    @staticmethod
    def _get_batch(env, size):
        s, x, y, p, c = env._get_random_batch(size, config.PRETRAIN_ZERO_PROB)

        q_c = np.full((size, config.TERMINAL_ACTIONS), config.REWARD_INCORRECT, dtype=np.float32)
        q_c[np.arange(size), y] = config.REWARD_CORRECT

        if config.USE_HPC:
            q_c[p == y, config.HPC_ACTION] = config.REWARD_CORRECT
            q_c[:, config.HPC_ACTION] -= c

        # make it cuda
        s   = torch.from_numpy(s).cuda()
        q_c = torch.from_numpy(q_c).cuda()

        return s, q_c

    def pretrain(self, env):
        test_s, test_q_c = self._get_batch(env, config.PRETRAIN_BATCH)

        last_loss = 9999.
        self.model.set_lr(config.PRETRAIN_LR)
        for i in range(config.PRETRAIN_EPOCHS):
            utils.print_progress(i, config.PRETRAIN_EPOCHS, step=100)

            # print loss
            if utils.is_time(i, 100):
                q = self.model(test_s)

                loss_c = self.model.get_loss_c(q, test_q_c)
                print("\nLoss: {}".format(loss_c.data.item()))

                if last_loss <= loss_c:     # stop early
                    break

                last_loss = loss_c

            s, q_c = self._get_batch(env, config.PRETRAIN_BATCH)
            self.model.train_c(s, q_c)

        self.model_.copy_weights(self.model, rho=1.0)

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
