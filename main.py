import pandas as pd
import numpy as np

import utils, json, random, torch, gc, types
import argparse
import sys
from config import config
import time
from pathlib import Path

# ==============================
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="dataset name")

parser.add_argument("--flambda", type=float, required=True, help="cost factor lambda")
parser.add_argument("--seed", type=int, default=None, help="random seed")

parser.add_argument(
    "--load_progress",
    type=str2bool,
    default=False,
    help="Whether to load a saved model.",
)
parser.add_argument(
    "--use_hpc",
    type=str2bool,
    default=True,
    help="Whether to use High Precision Classifier",
)
parser.add_argument(
    "--pretrain",
    type=str2bool,
    default=True,
    help="Whether to pretrain classification actions",
)
args = parser.parse_args()

config.init(args)
config.print_short()

timestamp = str(int(time.time()))

DATASET = args.dataset

OUTPUT_PATH = (
    Path.home()
    / "cwcf"
    / "output"
    / "drl"
    / str(DATASET)
    / 'flambda'+str(args.flambda)
    / "-".join(("drl", DATASET, str(args.flambda), timestamp))
)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# drl_stdout = str(OUTPUT_PATH / f"{DATASET}-drl-stdout-{timestamp}.log")
# drl_stderr = str(OUTPUT_PATH / f"{DATASET}-drl-stderr-{timestamp}.log")
#
# sys.stdout = open(drl_stdout, "w")
# sys.stderr = open(drl_stderr, "w")

print(f"Using dataset: {DATASET}")
print(f"Output Path: {OUTPUT_PATH}")

# ==============================
from agent import Agent
from brain import Brain
from env import Environment
from log import Log
from pool import Pool

# ==============================
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

# ==============================
if config.SEED:
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

# ==============================
data_trn = pd.read_pickle(config.DATA_FILE)
data_val = pd.read_pickle(config.DATA_VAL_FILE)
meta = pd.read_pickle(config.META_FILE)
hpc = pd.read_pickle(config.HPC_FILE)

feats = meta.index
costs = meta[config.META_COSTS]

data_trn[feats] = (data_trn[feats] - meta[config.META_AVG]) / meta[
    config.META_STD
]  # normalize
data_val[feats] = (data_val[feats] - meta[config.META_AVG]) / meta[
    config.META_STD
]  # normalize

print("Using", config.DATA_FILE, "with", len(data_trn), "samples.")
# ==============================
pool = Pool(config.POOL_SIZE)

env = Environment(data_trn, hpc["train"], costs)
brain = Brain(pool)
agent = Agent(env, pool, brain)
log_val = Log(data_val, hpc["validation"], costs, brain, OUTPUT_PATH, "val")
log_trn = Log(data_trn, hpc["train"], costs, brain, OUTPUT_PATH, "trn")

# ==============================
epoch_start = 0
lr_start = config.OPT_LR
avg_r = types.SimpleNamespace()

avg_r.trn_avg = []
avg_r.trn_run = []
avg_r.val_avg = []
avg_r.val_run = []
avg_r.trn_best = -999.0
avg_r.val_best = -999.0
avg_r.val_fails = 0

if not config.BLANK_INIT:
    print("Loading progress..")
    brain._load()

    with open(str(OUTPUT_PATH / "run.state"), "r") as file:
        save_data = json.load(file)

    epoch_start = save_data["epoch"]
    lr_start = save_data["lr"]
    avg_r = types.SimpleNamespace(**save_data["avg_r"])

# ======= PRETRAINING ==========
if config.PRETRAIN and config.BLANK_INIT:
    print("Pretraining..")
    brain.pretrain(env)
    brain._save(file="model_pretrained", filepath=OUTPUT_PATH)
# brain._load(file="model_pretrained")

# ==============================
agent.update_epsilon(epoch_start)
brain.update_epsilon(epoch_start)
brain.set_lr(lr_start)

# ==============================
print("Initializing pool..")
while pool.total < config.POOL_SIZE:
    agent.step()
    utils.print_progress(pool.total, config.POOL_SIZE, step=10)

# clear cache
gc.collect()
torch.cuda.empty_cache()

print("\nStarting..")

for epoch in range(epoch_start, config.MAX_TRAINING_EPOCHS + 1):
    # save progress
    if utils.is_time(epoch, config.SAVE_EPOCHS):
        brain._save(filepath=OUTPUT_PATH)
        save_data = {}
        save_data["epoch"] = epoch
        save_data["lr"] = brain.lr
        save_data["avg_r"] = avg_r.__dict__
        with open(str(OUTPUT_PATH/"run.state"), "w") as file:
            json.dump(save_data, file)

    # update exploration
    if utils.is_time(epoch, config.EPSILON_UPDATE_EPOCHS):
        agent.update_epsilon(epoch)
        brain.update_epsilon(epoch)

    # log
    if utils.is_time(epoch, config.LOG_EPOCHS):
        print("\nEpoch: {}/{}".format(epoch, config.MAX_TRAINING_EPOCHS))
        print(
            "Exploration e: {:.2f}, Target e: {:.2f}".format(
                agent.epsilon, brain.epsilon
            )
        )
        log_val.print_speed()

    if utils.is_time(epoch, config.LOG_PERF_EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()

        log_val.log_q()

        res_trn = log_trn.log_perf()
        res_val = log_val.log_perf()

        avg_r.val_run.append(res_val[0])
        avg_r.trn_run.append(res_trn[0])

    # evaluate training & validation error
    if utils.is_time(epoch, config.EVALUATE_STEPS):
        avg_r.trn_avg.append(np.mean(avg_r.trn_run))
        avg_r.trn_run = []
        avg_r.val_avg.append(np.mean(avg_r.val_run))
        avg_r.val_run = []

        # training error - lower LR if needed
        print("Training set averages over {} steps:".format(config.EVALUATE_STEPS))
        print(avg_r.trn_avg)

        trn_avg_last = avg_r.trn_avg[-1]
        if trn_avg_last <= avg_r.trn_best:
            print("Failed to improve on the training set, lowering learning-rate.")
            brain.lower_lr()

        avg_r.trn_best = trn_avg_last

        # validation error - early stop if failed 3 times
        print("Validation set averages over {} steps:".format(config.EVALUATE_STEPS))
        print(avg_r.val_avg)

        val_avg_last = avg_r.val_avg[-1]
        if val_avg_last > avg_r.val_best:
            avg_r.val_fails = 0
            avg_r.val_best = val_avg_last
            brain._save(file="model_best", filepath=OUTPUT_PATH)
        else:
            avg_r.val_fails += 1
            print(
                "Failed to improve on the validation set for {}-time.".format(
                    avg_r.val_fails
                )
            )
            if avg_r.val_fails >= config.VALIDATION_FAILS:
                print("Stopping...")
                break

    # TRAIN
    brain.train()

    for i in range(config.EPOCH_STEPS):
        agent.step()

# Log test performance
data_tst = pd.read_pickle(config.DATA_TEST_FILE)
data_tst[feats] = (data_tst[feats] - meta[config.META_AVG]) / meta[
    config.META_STD
]  # normalize

brain._load(file="model_best")
print("Performance on the best model:")
log_trn = Log(data_trn, hpc["train"], costs, brain, OUTPUT_PATH, "trn_best")
log_trn.log_perf()

log_val = Log(data_val, hpc["validation"], costs, brain, OUTPUT_PATH, "val_best")
log_val.log_perf()

log_tst = Log(data_tst, hpc["test"], costs, brain, OUTPUT_PATH, "tst_best")
log_tst.log_perf(histogram=True)

# # Close Log File
# sys.stdout.close()
# sys.stderr.close()
