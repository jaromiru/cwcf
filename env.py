import numpy as np

from consts import *
#==============================

LIN_ARRAY = np.arange(AGENTS)
#==============================
class Environment:
	def __init__(self, data, costs, ff):
		self.data_x = data.iloc[:, 0:-1].astype('float32').as_matrix()
		self.data_y = data.iloc[:,   -1].astype('int32').as_matrix()
		self.data_len = len(data)

		self.costs = costs.as_matrix()

		self.mask = np.zeros( (AGENTS, FEATURE_DIM) )
		self.x    = np.zeros( (AGENTS, FEATURE_DIM) )
		self.y    = np.zeros( AGENTS )

		self.ff = ff

	def reset(self):
		for i in range(AGENTS):
			self._reset(i)

		return self._get_state()

	def _reset(self, i):
		self.mask[i] = 0
		self.x[i], self.y[i] = self._generate_sample()

	def step(self, action):
		self.mask[LIN_ARRAY, action - CLASSES] = 1

		r = -self.costs[action - CLASSES] * self.ff

		for i in np.where(action < CLASSES)[0]:
			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
			self._reset(i)

		s_ = self._get_state()

		return (s_, r)

	def _generate_sample(self):
		idx = np.random.randint(0, self.data_len)

		x = self.data_x[idx]
		y = self.data_y[idx]

		return (x, y)

	def _get_state(self):
		x_ = self.x * self.mask
		x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32)
		return x_
		