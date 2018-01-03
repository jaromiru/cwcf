import numpy as np
import time, sys, utils

from consts import *

#==============================
class PerfAgent():
	def __init__(self, env, brain):
		self.env  = env
		self.brain = brain

		self.agents = self.env.agents

		self.done = np.zeros(self.agents, dtype=np.bool)
		self.total_r = np.zeros(self.agents)
		self.total_len = np.zeros(self.agents, dtype=np.int32)
		self.total_corr  = np.zeros(self.agents, dtype=np.int32)

		self.s = self.env.reset()

	def act(self, s):
		m = np.zeros((self.agents, ACTION_DIM))	# create max_mask
		m[:, CLASSES:] = s[:, FEATURE_DIM:]

		p = self.brain.predict_np(s) - MAX_MASK_CONST * m 	# select an action not considering those already performed
		a = np.argmax(p, axis=1)

		return a

	def step(self):
		a = self.act(self.s)
		s_, r, done = self.env.step(a)
		self.s = s_

		newly_finished = ~self.done & done
		self.done = self.done | done

		self.total_r   = self.total_r   + r * (newly_finished | ~done)
		self.total_len = self.total_len + ~done
		self.total_corr = self.total_corr + (r == REWARD_CORRECT) * newly_finished

	def run(self):
		while not np.all(self.done):
			# utils.print_progress(np.sum(self.done), self.agents, step=1)
			self.step()

		avg_r    = np.mean(self.total_r)
		avg_len  = np.mean(self.total_len)
		avg_corr = np.mean(self.total_corr)

		return avg_r, avg_len, avg_corr

#==============================
class PerfEnv:
	def __init__(self, data, costs, ff):
		data_val_idx = np.random.choice(len(data), LOG_PERF_VAL_SIZE)

		self.x = data.iloc[data_val_idx, 0:-1].astype('float32').as_matrix()
		self.y = data.iloc[data_val_idx,   -1].astype('int32').as_matrix()
		self.costs = costs.as_matrix()

		self.agents = LOG_PERF_VAL_SIZE
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
class Log:
	def __init__(self, data_val, costs, ff, brain):
		self.env = PerfEnv(data_val, costs, ff)
		self.brain = brain

		self.LOG_TRACKED_STATES = np.vstack(LOG_TRACKED_STATES).astype(np.float32)
		self.LEN = len(self.LOG_TRACKED_STATES)

		if BLANK_INIT:
			mode = "w"
		else:
			mode = "a"

		self.files = []
		for i in range(self.LEN):
			self.files.append( open("run_%d.dat" % i, mode) )

		self.perf_file = open("run_perf.dat", mode)

		self.time = 0

	def log(self):
		val = self.brain.predict_np(self.LOG_TRACKED_STATES)

		for i in range(self.LEN):
			w = val[i].data

			for k in w:
				self.files[i].write('%.4f ' % k)

			self.files[i].write('\n')
			self.files[i].flush()

	def print_speed(self):
		if self.time == 0:
			self.time = time.perf_counter()
			return

		now = time.perf_counter()
		elapsed = now - self.time
		self.time = now

		samples_processed = LOG_EPOCHS * EPOCH_STEPS * AGENTS
		updates_processed = LOG_EPOCHS
		updates_total = LOG_EPOCHS * BATCH_SIZE

		fps_smpl = samples_processed / elapsed
		fps_updt = updates_processed / elapsed
		fps_updt_t = updates_total / elapsed

		print("Perf.: {:.0f} smp/s, {:.1f} upd/s, {:.1f} upd_smp/s".format(fps_smpl, fps_updt, fps_updt_t))

	def log_perf(self):
		agent = PerfAgent(self.env, self.brain)
		avg_r, avg_len, avg_corr = agent.run()

		print("{:.3f} {:.3f} {:.3f}".format(avg_r, avg_len, avg_corr), file=self.perf_file, flush=True)