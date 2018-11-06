# Instructions:
# =============
# 1. Download http://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt into ../data/miniboone.dat
# 2. Remove fist row and leading spaces
# 3. Run this script

import pandas as pd
import numpy as np

COL_LABEL = '_label'

SEED = 998822
#---
np.random.seed(SEED)

data = pd.read_csv("../data/raw/miniboone.dat", header=None, sep=' +')

data[COL_LABEL] = 0
data.iloc[36500:][COL_LABEL] = 1

data = data[ data[0] > -900]

data.iloc[:, 0:-1] = data.iloc[:, 0:-1].astype('float32')
data.iloc[:,-1:  ] = data.iloc[:,-1:  ].astype('int32')

print(data.head())

TRAIN_SIZE = 45359
VAL_SIZE   = 19439
TEST_SIZE  = 64798

data = data.sample(frac=1)

data_train = data.iloc[0:TRAIN_SIZE]
data_val   = data.iloc[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
data_test   = data.iloc[TRAIN_SIZE+VAL_SIZE:]

print("Number of features:", data_train.shape[1] - 1)
print("Classes:", data_train.iloc[:, -1].unique())

print()
print("Total len:", data.shape[0])
print("----------")
print("Train len:", data_train.shape[0])
print("Val len:  ", data_val.shape[0])
print("Test len: ", data_test.shape[0])

data_train.to_pickle("../data/mb-train")
data_val.to_pickle("../data/mb-val")
data_test.to_pickle("../data/mb-test")

#--- prepare meta
idx = data.columns[:-1]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = data_train.mean()
meta['std'] = data_train.std()
meta['cost'] = 1.

meta = meta.astype('float32')

print()
print(meta)


meta.to_pickle("../data/mb-meta")