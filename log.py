import numpy as np
import time

from config import config

from agent import PerfAgent
from env import SeqEnvironment
from pathlib import Path


# ==============================
class Log:
    def __init__(self, data, hpc_p, costs, brain, log_name, output_path):
        self.data = data
        self.hpc_p = hpc_p
        self.costs = costs
        self.brain = brain

        self.LOG_TRACKED_STATES = np.array(config.LOG_TRACKED_STATES, dtype=np.float32)
        self.LEN = len(self.LOG_TRACKED_STATES)

        if log_name is None:
            raise ValueError("provide log name")
        else:
            self.log_name = log_name
        self.log_to_file = True

        self.OUTPUT_PATH = output_path

        # drl_stdout = str(OUTPUT_PATH / f"{DATASET}-hpc-stdout-{timestamp}.log")
        # drl_stderr = str(OUTPUT_PATH / f"{DATASET}-hpc-stderr-{timestamp}.log")
        # sys.stdout = open(drl_stdout, "w")
        # sys.stderr = open(drl_stderr, "w")
        # print(f"Using dataset: {DATASET}")
        # print(f"Output Path: {OUTPUT_PATH}")

        if self.log_to_file:
            if config.BLANK_INIT:
                mode = "w"
            else:
                mode = "a"

            self.files = []
            for i in range(self.LEN):
                self.files.append(
                    open(self.OUTPUT_PATH / f"run_{log_name}_{i}.dat", mode)
                )

            self.perf_file = open(self.OUTPUT_PATH / f"run_{log_name}_perf.dat", mode)

        self.time = 0

    def log_q(self):
        if self.log_to_file:
            q = self.brain.predict_np(self.LOG_TRACKED_STATES)

            for i in range(self.LEN):
                w = q[i].data

                for k in w:
                    self.files[i].write("%.4f " % k)

                self.files[i].write("\n")
                self.files[i].flush()

    def print_speed(self):
        if self.time == 0:
            self.time = time.perf_counter()
            return

        now = time.perf_counter()
        elapsed = now - self.time
        self.time = now

        samples_processed = config.LOG_EPOCHS * config.EPOCH_STEPS * config.AGENTS
        updates_processed = config.LOG_EPOCHS
        updates_total = config.LOG_EPOCHS * config.BATCH_SIZE

        fps_smpl = samples_processed / elapsed
        fps_updt = updates_processed / elapsed
        fps_updt_t = updates_total / elapsed

        print(
            "Perf.: {:.0f} gen_smp/s, {:.1f} upd/s, {:.1f} upd_steps/s".format(
                fps_smpl, fps_updt, fps_updt_t
            )
        )

    def log_perf(self, histogram=False):
        env = SeqEnvironment(self.data, self.hpc_p, self.costs)
        agent = PerfAgent(env, self.brain)

        # compute metrics
        _r = 0.0
        _fc = 0.0
        _len = 0.0
        _corr = 0.0
        _hpc = 0.0
        _lens = []
        _lens_hpc = []

        while True:
            # utils.print_progress(np.sum(self.done), self.agents, step=1)
            s, a, r, s_, done, info = agent.step()

            if np.all(done == -1):
                break

            finished = done == 1  # episode finished
            terminated = done == -1  # no more data

            _r += np.sum(r[~terminated])
            _fc += np.sum(r[~finished]) + np.sum(info["hpc_fc"])
            _corr += np.sum(info["corr"])
            _len += np.sum(~terminated)
            _hpc += np.sum(info["hpc"])

            if histogram:
                finished_hpc = finished * info["hpc"]
                finished_nohpc = finished * ~info["hpc"]

                _lens.append(info["eplen"][finished_nohpc])
                _lens_hpc.append(info["eplen"][finished_hpc])

        data_len = len(self.data)
        _r /= data_len
        _fc /= data_len * config.FEATURE_FACTOR * -1
        _corr /= data_len
        _len /= data_len
        _hpc /= data_len

        print(
            "{} R: {:.3f} | L: {:.3f} | FC: {:.3f} | HPC: {:.3f} | C: {:.3f}".format(
                self.log_name, _r, _len, _fc, _hpc, _corr
            )
        )

        if self.log_to_file:
            print(
                "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(_r, _len, _fc, _hpc, _corr),
                file=self.perf_file,
                flush=True,
            )

        if histogram:
            _lens = np.concatenate(_lens).flatten()
            _lens_hpc = np.concatenate(_lens_hpc).flatten()

            # print("Writing histogram...")
            with open(
                self.OUTPUT_PATH / f"run_{self.log_name}_histogram.dat", "w"
            ) as file:
                for x in _lens:
                    file.write("{} ".format(x))

            with open(
                self.OUTPUT_PATH / f"run_{self.log_name}_histogram_hpc.dat", "w"
            ) as file:
                for x in _lens_hpc:
                    file.write("{} ".format(x))

        return _r, _len, _fc, _hpc, _corr
