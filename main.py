import pandas as pd
import numpy as np

import utils, json, random, torch, gc, types
import argparse

from config import config
from enum import Enum

#==============================
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help="dataset name")
parser.add_argument('target', type=float, help="target cost / lambda")

parser.add_argument('-target_type', type=str, choices=['lambda', 'cost'], default='lambda', help="Whether to target lambda or average cost")
parser.add_argument('-seed', type=int, default=None, help="random seed")

parser.add_argument('-load_progress', type=str2bool, default=False, help="Whether to load a saved model.")
parser.add_argument('-use_hpc', type=str2bool, default=False, help="Whether to use High Precision Classifier")
parser.add_argument('-pretrain', type=str2bool, default=True, help="Whether to pretrain classification actions")

parser.add_argument('-hard_budget', type=float, default=np.inf, help="Compute with hard budget")
parser.add_argument('-missing_grad', type=str, choices=['full', 'sparse'], default='sparse', help="How to handle the missing values during learning")

# data = convert to balanced dataset; data_gradient = train with balanced weights, but for the original dataset
parser.add_argument('-reweight', choices=['no', 'data', 'data_gradient'], default='no', help="Whether to rebalance the dataset")

parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help="Which device to use")
parser.add_argument('-forcerun', type=str2bool, default=False, help="Force run until config.MAX_TRAINING_EPOCHS")

args = parser.parse_args()

print(args)
config.init(args)
config.print_short()

#==============================
from agent import Agent
from brain import Brain
from env import Environment
from log import Log
from pool import Pool

#==============================
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

#==============================
if config.SEED:
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)

#==============================
data_trn = pd.read_pickle(config.DATA_FILE)
data_val = pd.read_pickle(config.DATA_VAL_FILE)
meta = pd.read_pickle(config.META_FILE)

if config.USE_HPC:
    hpc  = pd.read_pickle(config.HPC_FILE)
else:   
    zeroObj = utils.ZeroObj()
    hpc = {'train': zeroObj, 'validation': zeroObj, 'test': zeroObj}

feats = meta.index
costs = meta[config.META_COSTS]
costs_sum = np.sum(costs)

data_trn[feats] = (data_trn[feats] - meta[config.META_AVG]) / meta[config.META_STD]  # normalize
data_val[feats] = (data_val[feats] - meta[config.META_AVG]) / meta[config.META_STD]  # normalize

print("Using", config.DATA_FILE, "with", len(data_trn), "samples.")
#==============================
pool  = Pool(config.POOL_SIZE)

env   = Environment(data_trn, hpc['train'], costs)
brain = Brain(pool)
agent = Agent(env, pool, brain)
log_val = Log(data_val, hpc['validation'], costs, brain, "val")
log_trn = Log(data_trn, hpc['train'], costs, brain, "trn")

#==============================
epoch_start = 0
lr_start = config.OPT_LR

# average d_lagrangian
avg_l = types.SimpleNamespace()
avg_l.trn_avg = []
avg_l.trn_run = []
avg_l.val_avg = []
avg_l.val_run = []
avg_l.trn_lst = 0.
avg_l.val_lst = 0.

if not config.BLANK_INIT:
    print("Loading progress..")
    brain._load()

    with open('run.state', 'r') as file:
        save_data = json.load(file)

    epoch_start = save_data['epoch']
    lr_start = save_data['lr']
    avg_l = types.SimpleNamespace(**save_data['avg_l'])

    if args.target_type == 'cost':
        config.FEATURE_FACTOR = save_data['lmb']
else:
    # truncate files
    open('run_{}_lagrangian.dat'.format("trn"), "w").close()
    open('run_{}_lagrangian.dat'.format("val"), "w").close()


#======= PRETRAINING ==========
if config.PRETRAIN and config.BLANK_INIT:
    print("Pretraining..")
    brain.pretrain(env)
    brain._save(file="model_pretrained")
# brain._load(file="model_pretrained")

#==============================
agent.update_epsilon(epoch_start)
brain.update_epsilon(epoch_start)
brain.set_lr(lr_start)
lr_lmb = config.OPT_LAMBDA_LR / (1 + epoch_start // config.OPT_LR_STEPS)
print("Lambda LR: {:.2e}".format(lr_lmb))        

#==============================
print("Initializing pool..")
while pool.total < config.POOL_SIZE:
    agent.step()
    utils.print_progress(pool.total, config.POOL_SIZE, step=10)

# clear cache
gc.collect()
torch.cuda.empty_cache()

lambda_last_grad = 0.
old_params = np.concatenate([brain.model.param_array(), [config.FEATURE_FACTOR]])

print("\nStarting..")
for epoch in range(epoch_start, config.MAX_TRAINING_EPOCHS + 1):
    # save progress
    if utils.is_time(epoch, config.SAVE_EPOCHS):
        brain._save()

        save_data = {}
        save_data['epoch'] = epoch
        save_data['lr'] = brain.lr
        save_data['lmb'] = config.FEATURE_FACTOR
        save_data['avg_l'] = avg_l.__dict__

        with open('run.state', 'w') as file:
            json.dump(save_data, file)
    
    # update exploration
    if utils.is_time(epoch, config.EPSILON_UPDATE_EPOCHS):
        agent.update_epsilon(epoch)
        brain.update_epsilon(epoch)

    # log
    if utils.is_time(epoch, config.LOG_EPOCHS):
        gc.collect()
        torch.cuda.empty_cache()

        print("\nEpoch: {}/{}".format(epoch, config.MAX_TRAINING_EPOCHS))
        print("Exploration e: {:.2f}, Target e: {:.2f}".format(agent.epsilon, brain.epsilon))
        log_val.print_speed()

        log_trn.log_q()
        log_val.log_q()

        _, _, trn_cost, _, trn_corr = log_trn.log_perf()
        _, _, val_cost, _, val_corr = log_val.log_perf()

        # we technically don't need this, when target=lambda...
        trn_l = (1 - trn_corr) + config.FEATURE_FACTOR * (trn_cost - config.TARGET_COST)
        val_l = (1 - val_corr) + config.FEATURE_FACTOR * (val_cost - config.TARGET_COST)

        new_params = np.concatenate([brain.model.param_array(), [config.FEATURE_FACTOR]])
        param_d    = old_params - new_params
        param_norm = np.sqrt(param_d.dot(param_d))
        old_params = new_params

        trn_dl = np.abs(trn_l - avg_l.trn_lst) / param_norm
        val_dl = np.abs(val_l - avg_l.val_lst) / param_norm

        print("DL: {}".format(trn_dl))

        avg_l.trn_lst = trn_l
        avg_l.val_lst = val_l

        avg_l.trn_run.append([trn_dl, trn_corr])
        avg_l.val_run.append([val_dl, val_corr])

    # change lambda is targeting cost
    if config.TARGET_TYPE == 'cost' and utils.is_time(epoch, config.OPT_LAMBDA_EPOCHS):
        grad = config.OPT_LAMBDA_GAMMA * lambda_last_grad + lr_lmb * (trn_cost - config.TARGET_COST) / costs_sum
        lambda_last_grad = grad
        
        config.FEATURE_FACTOR = np.clip( config.FEATURE_FACTOR + grad, 0, 1)
        print("lambda: {:.5f}, grad: {:.5f}".format(config.FEATURE_FACTOR, grad))        

        # debug
        with open('run_{}_lagrangian.dat'.format("trn"), 'a') as file:
            print("{} {} {}".format(trn_cost, trn_corr, config.FEATURE_FACTOR), file=file)

        with open('run_{}_lagrangian.dat'.format("val"), 'a') as file:
            print("{} {} {}".format(val_cost, val_corr, config.FEATURE_FACTOR), file=file)


    # evaluate training & validation error
    if utils.is_time(epoch, config.EVALUATE_STEPS):
        avg_l.trn_avg.append( np.mean(avg_l.trn_run, axis=0).tolist() )
        avg_l.trn_run = []
        avg_l.val_avg.append( np.mean(avg_l.val_run, axis=0).tolist() )
        avg_l.val_run = []

        print("Trn (lag acc):", " ".join(["({:0.5f} {:0.4f})".format(x[0], x[1]) for x in avg_l.trn_avg]))
        print("Val (lag acc):", " ".join(["({:0.5f} {:0.4f})".format(x[0], x[1]) for x in avg_l.val_avg]))

    # check for terminating conditions
    if utils.is_time(epoch, config.EVALUATE_STEPS) and (epoch // config.EVALUATE_STEPS) >= max(config.ACCURACY_CHECK, config.LAGRANGIAN_CHECK_TIMES):
        acc_ok = True
        lag_ok = True

        # check if accuracy improved (validation set)
        acc = [x[1] for x in avg_l.val_avg[-config.ACCURACY_CHECK:]]
        if acc[0] >= np.max(acc[1:]):
            acc_ok = False  # failed to improve  

        # check if lagrangian is stable (training set)   
        lag = [x[0] for x in avg_l.trn_avg[-config.ACCURACY_CHECK:]]
        if lag[0] <= np.min(lag[1:]):
            lag_ok = False  # failed to improve  

        # lag = [x[0] for x in avg_l.trn_avg[-config.LAGRANGIAN_CHECK_TIMES:]]
        # if np.max(lag) <= config.LAGRANGIAN_CHECK_TRESHOLD:
        #     lag_ok = False

        print("Acc improved: {}, Lagrangian unstable: {}".format(acc_ok, lag_ok))
        if (not acc_ok) and (not lag_ok) and (not args.forcerun):
            break;

    if utils.is_time(epoch, config.OPT_LR_STEPS):    
        ep = epoch // config.OPT_LR_STEPS

        # --- 1/T schedule
        # lr_net = config.OPT_LR / (1 + ep)
        # lr_lmb = config.OPT_LAMBDA_LR / (1 + ep)

        # --- * (0.5^T) schedule
        lr_net = config.OPT_LR * (config.OPT_LR_FACTOR ** ep)
        lr_lmb = config.OPT_LAMBDA_LR * (config.OPT_LR_FACTOR ** ep)
        
        brain.set_lr(lr_net)
        print("Lambda LR: {:.2e}".format(lr_lmb))

    # TRAIN
    brain.train()

    for i in range(config.EPOCH_STEPS):
        agent.step()

# Log test performance
data_tst = pd.read_pickle(config.DATA_TEST_FILE)
data_tst[feats] = (data_tst[feats] - meta[config.META_AVG]) / meta[config.META_STD]   # normalize

# print("Test performance on the last model:")
# log_tst = Log(data_tst, hpc['test'], costs, brain, "tst_last")
# log_tst.log_perf()

# brain._load(file='model_best')
print("Performance on the best model:")
log_trn = Log(data_trn, hpc['train'], costs, brain, "trn_best")
log_trn.log_perf()

log_val = Log(data_val, hpc['validation'], costs, brain, "val_best")
log_val.log_perf()

log_tst = Log(data_tst, hpc['test'], costs, brain, "tst_best")
log_tst.log_perf(histogram=True, verbose=True)