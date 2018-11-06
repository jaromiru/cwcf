# Instructions:
# =============
# 1. Extract https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz into ../data/covtype.dat
# 2. Run this script

import pandas as pd
import numpy as np

SEED = 998823
#---
np.random.seed(SEED)

data = pd.read_csv("../data/raw/covtype.dat", header=None, sep=',')
print("Loaded", len(data), "rows.")

data.iloc[:, 0:-1] = data.iloc[:, 0:-1].astype('float32')
data.iloc[:,-1:  ] = data.iloc[:,-1:  ].astype('int32') - 1

print(data.head())

TRAIN_SIZE = 200000
VAL_SIZE   =  81012
TEST_SIZE  = 300000

# TRAIN_SIZE = 36603
# VAL_SIZE   = 15688
# TEST_SIZE  = 58101

data = data.sample(frac=1)

data_train = data.iloc[0:TRAIN_SIZE]
data_val   = data.iloc[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
data_test   = data.iloc[TRAIN_SIZE+VAL_SIZE:TRAIN_SIZE+VAL_SIZE+TEST_SIZE]

print("Number of features:", data_train.shape[1] - 1)
print("Classes:", data_train.iloc[:, -1].unique())

print()
print("Total len:", data.shape[0])
print("----------")
print("Train len:", data_train.shape[0])
print("Val len:  ", data_val.shape[0])
print("Test len: ", data_test.shape[0])

data_train.to_pickle("../data/forest-train")
data_val.to_pickle("../data/forest-val")
data_test.to_pickle("../data/forest-test")

#--- prepare meta
idx = data.columns[:-1]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = data_train.mean()
meta['std'] = data_train.std()
meta['absmax'] = data_train.abs().max()

#--- forest binary features
# meta.loc[10:, 'avg'] = 0
# meta.loc[10:, 'std'] = 1.0

meta['cost'] = 1.

meta = meta.astype('float32')

print(meta)
meta.to_pickle("../data/forest-meta")