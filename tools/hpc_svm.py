""" Computes probabilities for HPC model """
import pickle

from sklearn.svm import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np

import argparse
from pathlib import Path
import sys
import random
import time

data_path = Path.home() / "cwcf" / "data"
timestamp = str(int(time.time()))

hpc_stdout = Path.home() / "cwcf" / "logs" / f"hpc_svm{timestamp}_stdout.log"
hpc_stderr = Path.home() / "cwcf" / "logs" / f"hpc_svm{timestamp}_stderr.log"
hpc_stderr.parent.mkdir(parents=True, exist_ok=True)

sys.stdout = open(str(hpc_stdout), "w")
sys.stderr = open(str(hpc_stderr), "w")

# ----------------

META_AVG = "avg"
META_STD = "std"

# ----------------

def get_full_rbf_svm_clf(train_x, train_y, c_range=None, gamma_range=None):
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    grid = GridSearchCV(
        SVC(cache_size=1024), param_grid=param_grid, cv=cv, n_jobs=20, verbose=51
    )
    grid.fit(train_x, train_y)

    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

    scores = grid.cv_results_["mean_test_score"].reshape(len(c_range), len(gamma_range))
    print("Scores:")
    print(scores)

    print("c_range:", c_range)
    print("gamma_range:", gamma_range)

    c_best = grid.best_params_["C"]
    gamma_best = grid.best_params_["gamma"]

    clf = SVC(C=c_best, gamma=gamma_best, verbose=51)
    return grid, clf


# ----------------
def prep(data):
    data[feats] = (data[feats] - meta[META_AVG]) / meta[META_STD]  # normalize
    data.fillna(0, inplace=True)  # impute NaNs with mean=0

    if "_count" in data.columns:
        data.drop("_count", axis=1, inplace=True)

    data_x = data.iloc[:, 0:-1].astype("float32").values
    data_y = data.iloc[:, -1].astype("int32").values

    return data_x, data_y


# ----------------

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", required=True, help="dataset name")
parser.add_argument("-svmgamma", type=float, help="SVM gamma parameter")
parser.add_argument("-svmc", type=float, help="SVM C parameter")

args = parser.parse_args()


DATASET = args.dataset

DATA_FILE = str(data_path / (DATASET + "-train"))
VAL_FILE = str(data_path / (DATASET + "-val"))
TEST_FILE = str(data_path / (DATASET + "-test"))
META_FILE = str(data_path / (DATASET + "-meta"))
HPC_FILE = str(data_path / (DATASET + "-hpc"))
OUTPUT_PATH = Path.home() / "cwcf" / "output"
OUTPUT_PATH / timestamp.mkdir(parents=True, exists_ok=True)

print("Using dataset", DATASET)
# ----------------

data_train = pd.read_pickle(DATA_FILE)
data_val = pd.read_pickle(VAL_FILE)
data_test = pd.read_pickle(TEST_FILE)
meta = pd.read_pickle(META_FILE)

feats = meta.index

train_x, train_y = prep(data_train)
val_x, val_y = prep(data_val)
test_x, test_y = prep(data_test)

if args.svmgamma is not None and args.svmc is not None:
    grid, clf = SVC(C=args.svmc, gamma=args.svmgamma, cache_size=4096)
else:
    print("Searching for hyperparameters...")
    c_range = np.logspace(-3, 3, 5)
    gamma_range = np.logspace(-5, 1, 5)
    grid, clf = get_full_rbf_svm_clf(
        train_x, train_y, c_range=c_range, gamma_range=gamma_range
    )

print("Training...")
clf.fit(train_x, train_y)

# ----------------
print("Trn score:  {:.4f}".format(clf.score(train_x, train_y)))
print("Val score:  {:.4f}".format(clf.score(val_x, val_y)))
print("Tst score:  {:.4f}".format(clf.score(test_x, test_y)))

# ----------------
print("\nSaving...")
train_p = clf.predict(train_x)
val_p = clf.predict(val_x)
test_p = clf.predict(test_x)

data_p = pd.DataFrame(
    data=[train_p, val_p, test_p], index=["train", "validation", "test"]
).transpose()
data_p.to_pickle(HPC_FILE)
data_p.to_pickle(str(OUTPUT_PATH / timestamp / f"{DATASET}-hpc-{timestamp}"))

with open(str(OUTPUT_PATH / timestamp / f"{DATASET}-hpc-clf-{timestamp}"), "wb") as f:
    pickle.dump(clf, f)

with open(str(OUTPUT_PATH / timestamp / f"{DATASET}-hpc-grid-{timestamp}"), "wb") as f:
    pickle.dump(grid, f)

# Close Log File
sys.stdout.close()
sys.stderr.close()
