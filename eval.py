import pandas as pd
import argparse

from config import config

#==============================
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help="dataset name")

parser.add_argument('--flambda', type=float, required=True, help="cost factor lambda")
parser.add_argument('--use_hpc', type=str2bool, default=True, help="Whether to use High Precision Classifier")
args = parser.parse_args()

args.seed = None
args.load_progress = True # will append to log files
args.pretrain = False

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
hpc  = pd.read_pickle(config.HPC_FILE)

feats = meta.index
costs = meta[config.META_COSTS]

data_trn[feats] = (data_trn[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize
data_val[feats] = (data_val[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize
data_tst[feats] = (data_tst[feats] - meta[config.META_AVG]) / meta[config.META_STD]		# normalize

#==============================
print("Evaluating dataset:", args.dataset)

brain = Brain(None)
brain._load(file='model_best')

print("Performance on the best model:")
log_trn = Log(data_trn, hpc['train'], costs, brain, "trn_best")
log_trn.log_perf()

log_val = Log(data_val, hpc['validation'], costs, brain, "val_best")
log_val.log_perf()

log_tst = Log(data_tst, hpc['test'], costs, brain, "tst_best")
log_tst.log_perf(histogram=True)