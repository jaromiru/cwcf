import numpy as np
import torch

from consts import *

class Pool():
	def __init__(self, size):
		self.data_s  = torch.FloatTensor(size, STATE_DIM)
		self.data_a  = torch.LongTensor(size, 1)
		self.data_r  = torch.FloatTensor(size, 1)
		self.data_s_ = torch.FloatTensor(size, STATE_DIM)

		self.idx  = 0
		self.size = size

	def put(self, x):
		s, a, r, s_ = x
		size = len(s)

		self.data_s [self.idx:self.idx+size] = torch.from_numpy(s)
		self.data_a [self.idx:self.idx+size] = torch.from_numpy(a)	
		self.data_r [self.idx:self.idx+size] = torch.from_numpy(r)
		self.data_s_[self.idx:self.idx+size] = torch.from_numpy(s_)

		self.idx = (self.idx + size) % self.size

	def sample(self, size):
		idx = torch.from_numpy(np.random.choice(self.size, size)).cuda()
		return self.data_s[idx], self.data_a[idx], self.data_r[idx], self.data_s_[idx]

	def cuda(self):
		self.data_s  = self.data_s.cuda() 
		self.data_a  = self.data_a.cuda() 
		self.data_r  = self.data_r.cuda() 
		self.data_s_ = self.data_s_.cuda()