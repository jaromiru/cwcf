""" Creates a random HPC predictions """

from sklearn.svm import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np

import argparse

# ----------------
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", required=True, help="dataset name")
parser.add_argument("-classes", required=True, help="number of classes")

args = parser.parse_args()

DATASET = args.dataset

DATA_FILE = "../../data/" + DATASET + "-train"
VAL_FILE = "../../data/" + DATASET + "-val"
TEST_FILE = "../../data/" + DATASET + "-test"
META_FILE = "../../data/" + DATASET + "-meta"
HPC_FILE = "../../data/" + DATASET + "-hpc-fake"

# ----------------
print("Using dataset", DATASET)
data_train = pd.read_pickle(DATA_FILE)
data_val = pd.read_pickle(VAL_FILE)
data_test = pd.read_pickle(TEST_FILE)

# ----------------
print("\nSaving...")
train_p = np.random.randint(0, args.classes, size=data_train.shape[0])
val_p = np.random.randint(0, args.classes, size=data_val.shape[0])
test_p = np.random.randint(0, args.classes, size=data_test.shape[0])

data_p = pd.DataFrame(
    data=[train_p, val_p, test_p], index=["train", "validation", "test"]
).transpose()
data_p.to_pickle(HPC_FILE)
