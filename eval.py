import pandas as pd
import numpy as np
import argparse

from config import config
import utils

#==============================
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('-load', action='store_const', const=True, help="try to load parameters from run.out file")

parser.add_argument('-dataset', type=str, default=None, help="dataset name")
parser.add_argument('-target', type=float, default=0, help="target cost / lambda")

parser.add_argument('-target_type', type=str, choices=['lambda', 'cost'], default='lambda', help="Whether to target lambda or average cost")
parser.add_argument('-use_hpc', type=str2bool, default=False, help="Whether to use High Precision Classifier")
parser.add_argument('-hard_budget', type=float, default=np.inf, help="Compute with hard budget")
parser.add_argument('-missing_grad', type=str, choices=['full', 'sparse'], default='sparse', help="How to handle the missing values during learning")
parser.add_argument('-reweight', choices=['no', 'data', 'data_gradient'], default='no', help="Whether to rebalance the dataset")
parser.add_argument('-device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto', help="Which device to use")

parser.add_argument('-set', nargs='+', default=['trn', 'val', 'tst'], help="Evaluate on these sets only (trn,val,tst).")

args = parser.parse_args()

args.seed = None
args.load_progress = True # will append to log files
args.pretrain = False

if args.load:
    with open('run.out', 'r') as f:
        # load
        line = f.readline()

        # potentially hazardous!
        from argparse import Namespace
        inf = np.inf
        new_args = eval(line)

        new_args.load_progress = True  # will append to log files
        print(new_args)
        config.init(new_args)

else: 
    print(args)
    config.init(args)

config.print_short()

#==============================
from brain import Brain
from log import Log

#==============================
data_trn = pd.read_pickle(config.DATA_FILE)
data_val = pd.read_pickle(config.DATA_VAL_FILE)
data_tst = pd.read_pickle(config.DATA_TEST_FILE)
meta = pd.read_pickle(config.META_FILE)

if config.USE_HPC:
    hpc  = pd.read_pickle(config.HPC_FILE)
else:   
    zeroObj = utils.ZeroObj()
    hpc = {'train': zeroObj, 'validation': zeroObj, 'test': zeroObj}

feats = meta.index
costs = meta[config.META_COSTS]

data_trn[feats] = (data_trn[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize
data_val[feats] = (data_val[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize
data_tst[feats] = (data_tst[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize

#==============================
print("Evaluating dataset:", config.dataset)

brain = Brain(None)
brain._load(file='model')

print("Performance on the last model:")

if 'trn' in args.set:
	log_trn = Log(data_trn, hpc['train'], costs, brain, "trn_best")
	log_trn.log_perf()

if 'val' in args.set:
	log_val = Log(data_val, hpc['validation'], costs, brain, "val_best")
	log_val.log_perf()

if 'tst' in args.set:
	log_tst = Log(data_tst, hpc['test'], costs, brain, "tst_best")
	log_tst.log_perf(histogram=True, verbose=True)
