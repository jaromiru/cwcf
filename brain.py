import numpy as np

import torch
from torch.autograd import Variable

import sys

from consts import *
from net import Net

#==============================
class Brain:
	def __init__(self, pool):
		self.pool = pool

		self.model  = Net()
		self.model_ = Net()

		print("Network architecture:\n"+str(self.model))

	def _load(self):
		self.model.load_state_dict( torch.load("model") )
		self.model_.load_state_dict( torch.load("model_") )

	def _save(self):
		torch.save(self.model.state_dict(), "model")
		torch.save(self.model_.state_dict(), "model_")

	def predict_pt(self, s, target):
		s = Variable(s)

		if target:
			return self.model_(s).data
		else:
			return self.model(s).data

	def predict_np(self, s, target=False):
		s = torch.from_numpy(s).cuda()
		res = self.predict_pt(s, target)
		return res.cpu().numpy()

	def train(self):
		s, a, r, s_ = self.pool.sample(BATCH_SIZE)

		# extract the mask
		m_ = torch.FloatTensor(BATCH_SIZE, ACTION_DIM).zero_().cuda()
		m_[:, CLASSES:] = s_[:, FEATURE_DIM:]

		# compute
		q_current = self.predict_pt(s_, target=False) - (MAX_MASK_CONST * m_) # masked actions do not influence the max
		q_target  = self.predict_pt(s_, target=True)

		_, amax = q_current.max(1, keepdim=True)
		q_ = q_target.gather(1, amax)

		q_[ a < CLASSES ] = 0
		q_ = q_ + r

		# bound the values to theoretical q function range
		q_.clamp_(-1, 0)

		self.model.train_network(s, a, q_)
		self.model_.copy_weights(self.model)

	def update_lr(self, epoch):
		lr = OPT_LR * (LR_SC_FACTOR ** (epoch // LR_SC_EPOCHS))
		lr = max(lr, LR_SC_MIN)

		self.model.set_lr(lr)
		print("Setting LR:", lr)
