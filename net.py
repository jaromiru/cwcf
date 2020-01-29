from config import config

import torch
import torch.nn.functional as F

import numpy as np
#==============================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        in_nn  = config.FEATURE_DIM * 2
        out_nn = config.NN_FC_DENSITY

        self.l_fc = []
        for i in range(config.NN_HIDDEN_LAYERS):
            layer = torch.nn.Linear(in_nn, out_nn)
            in_nn = out_nn

            self.l_fc.append(layer)
            self.add_module("l_fc_"+str(i), layer)

        # dueling architecture
        self.l_v = torch.nn.Linear(in_nn, 1)
        self.l_a = torch.nn.Linear(in_nn, config.ACTION_DIM)

        self.opt = torch.optim.Adam(self.parameters(), lr=config.OPT_LR, weight_decay=config.OPT_L2)
        # self.opt = torch.optim.RMSprop(self.parameters(), lr=OPT_LR, alpha=OPT_ALPHA, weight_decay=OPT_L2)

        self.loss_mse   = torch.nn.MSELoss()
        self.loss_cross = torch.nn.CrossEntropyLoss()

        if config.DEVICE == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.DEVICE)

        self.to(self.device)

    def forward(self, batch):
        flow = batch.view(-1, config.FEATURE_DIM * 2).to(self.device)

        for l in self.l_fc:
            flow = F.relu(l(flow))

        v = self.l_v(flow)
        a = self.l_a(flow)

        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def copy_weights(self, other, rho=config.TARGET_RHO):
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def param_array(self):
        return np.concatenate([x.data.cpu().numpy().flatten() for x in self.parameters()])

    def get_loss_a(self, q, a, q_a_target, w):
        ''' Returns a loss for the specified actions '''
        q_a_pred = q.gather(1, a).reshape(-1)
        loss_q_a = torch.mean( w * (q_a_pred - q_a_target) ** 2 )
        # loss_q_a = self.loss_mse(q_a_pred, q_a_target)

        return loss_q_a

    def get_loss_c(self, q, q_c_target, w):
        ''' Returns a loss for all classification actions + hpc '''
        q_c = q[:, :config.TERMINAL_ACTIONS]

        weighted_loss = w * torch.mean((q_c - q_c_target) ** 2, dim=1)
        loss_q_c = torch.mean(weighted_loss)
        # loss_q_c = self.loss_mse(q_c, q_c_target)

        return loss_q_c

    def get_loss_f(self, f_pred, f_target):
        ''' Returns a loss for feature prediction '''
        loss_f = self.loss_mse(f_pred, f_target)
        return loss_f

    def get_loss_y(self, y_pred, y_target):
        ''' Return a cross-entropy loss for class prediction '''
        loss_y   = self.loss_cross(y_pred, y_target)

        return loss_y

    # def train_s(self, s, q_c_target, f_target, y_target):
    #     q, f, y = self(s)

    #     loss_q_c = self.get_loss_c(q, q_c_target)
    #     loss_f   = self.get_loss_f(f, f_target)
    #     loss_y   = self.get_loss_y(y, y_target)

    #     loss = loss_q_c # + loss_f + loss_y
    #     self._perform_step(loss)

    def train_c(self, s, q_c_target, w):
        q = self(s)
        loss = self.get_loss_c(q, q_c_target, w)
        self._perform_step(loss)

    def train_pred(self, q, a, q_a_target, w):
        loss = self.get_loss_a(q, a, q_a_target, w)
        self._perform_step(loss)

    def _perform_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), config.OPT_MAX_NORM)
        self.opt.step()

    def set_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
