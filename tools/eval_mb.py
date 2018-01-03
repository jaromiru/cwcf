import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import aux.utils as utils
#==============================

DATA_FILE = "../data/mb-test"
META_FILE = "../data/mb-meta"
MODEL_FILE = "model"

META_COSTS = 'cost'
META_AVG   = 'avg'
META_STD   = 'std'

NN_FC_DENSITY = 128
NN_HIDDEN_LAYERS = 3

FEATURE_FACTOR   =   0.001
REWARD_CORRECT   =   0
REWARD_INCORRECT =  -1	

CLASSES = 2
FEATURE_DIM = 50
STATE_DIM = FEATURE_DIM * 2
ACTION_DIM = FEATURE_DIM + CLASSES

MAX_MASK_CONST = 1.e6
CHUNK_SIZE = 1000

#==============================
class PerfAgent():
	def __init__(self, env, brain):
		self.env  = env
		self.brain = brain

		self.agents = self.env.agents

		self.done = np.zeros(self.agents, dtype=np.bool)

		self.total_r = np.zeros(self.agents)
		self.total_corr  = np.zeros(self.agents, dtype=np.int32)
		self.total_len  = np.zeros(self.agents, dtype=np.int32)

		self.s = self.env.reset()

	def act(self, s):
		m = np.zeros((self.agents, ACTION_DIM))	# create max_mask
		m[:, CLASSES:] = s[:, FEATURE_DIM:]

		p = self.brain.predict(s) - MAX_MASK_CONST * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done
		self.done = self.done | done

		self.total_len = self.total_len + ~done	# classification action not counted
		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_corr = self.total_corr + (r == REWARD_CORRECT) * newly_finished

	def run(self):
		while not np.all(self.done):
			self.step()

		avg_r    = np.sum(self.total_r)
		avg_len  = np.sum(self.total_len)
		avg_corr = np.sum(self.total_corr)

		return avg_r, avg_len, avg_corr

#==============================
class PerfEnv:
	def __init__(self, data_x, data_y, costs, ff):
		self.x = data_x
		self.y = data_y
		self.costs = costs.as_matrix()

		self.agents = len(data_x)
		self.lin_array = np.arange(self.agents)

		self.ff = ff

	def reset(self):
		self.mask = np.zeros( (self.agents, FEATURE_DIM) )
		self.done = np.zeros( self.agents, dtype=np.bool )

		return self._get_state()

	def step(self, action):
		self.mask[self.lin_array, action - CLASSES] = 1

		r = -self.costs[action - CLASSES] * self.ff

		for i in np.where(action < CLASSES)[0]:
			r[i] = REWARD_CORRECT if action[i] == self.y[i] else REWARD_INCORRECT
			self.done[i] = 1

		s_ = self._get_state()

		return (s_, r, self.done)

	def _get_state(self):
		x_ = self.x * self.mask
		x_ = np.concatenate( (x_, self.mask), axis=1 ).astype(np.float32)
		return x_

#==============================
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		in_nn  = STATE_DIM
		out_nn = NN_FC_DENSITY

		self.l_fc = []
		for i in range(NN_HIDDEN_LAYERS):
			l = torch.nn.Linear(in_nn, out_nn)
			in_nn = out_nn

			self.l_fc.append(l)
			self.add_module("l_fc_"+str(i), l)

		self.l_out_q_val = torch.nn.Linear(in_nn, ACTION_DIM)		# q-value prediction
		# self.cuda()

	def forward(self, batch):
		flow = batch

		for l in self.l_fc:
			flow = F.relu(l(flow))

		a_out_q_val = self.l_out_q_val(flow)

		return a_out_q_val

#==============================
class Brain:
	def __init__(self):
		self.model  = Net()
		print("Network architecture:\n"+str(self.model))

	def _load(self):
		self.model.load_state_dict( torch.load(MODEL_FILE, map_location={'cuda:0': 'cpu'}) )

	def predict(self, s):
		# s = Variable(torch.from_numpy(s).cuda())
		s = Variable(torch.from_numpy(s))
		res = self.model(s)
		# return res.data.cpu().numpy()
		return res.data.numpy()

#==============================
data = pd.read_pickle(DATA_FILE)
meta = pd.read_pickle(META_FILE)

feats = meta.index
costs = meta[META_COSTS]

data[feats] = (data[feats] - meta[META_AVG]) / meta[META_STD]	# normalize
# data = data[0:10000]

data_x   = data.iloc[:, 0:-2].astype('float32').as_matrix()
data_y   = data.iloc[:,   -2].astype('int32').as_matrix()
data_len = len(data)

#==============================
brain = Brain()
brain._load()

confidence = np.zeros(data_len)
avg_r = 0
avg_len = 0
avg_corr = 0
idx = 0

while idx < data_len:
	utils.print_progress(idx, data_len, step=CHUNK_SIZE)

	last = min(idx + CHUNK_SIZE, data_len)

	data_batch_x = data_x[idx:last]
	data_batch_y = data_y[idx:last]

	env = PerfEnv(data_batch_x, data_batch_y, costs, FEATURE_FACTOR)
	agent = PerfAgent(env, brain)

	_r, _len, _corr = agent.run()
	avg_r += _r
	avg_len += _len
	avg_corr += _corr

	idx += CHUNK_SIZE

avg_len  /= data_len
avg_r    /= data_len
avg_corr /= data_len

print("R: ", avg_r)
print("Correct:", avg_corr)
print("Length", avg_len)
