# Instructions:
# =============
# 1. Collect from https://archive.ics.uci.edu/ml/datasets/yeast 
# 2. Run this script

import pandas as pd
import numpy as np

SEED = 998823

#---
np.random.seed(SEED)

data = pd.read_csv("../data/raw/yeast.data", header=None, sep=' +')

data = data.drop(0, axis=1)
data.columns = range(data.shape[1])

# categorize labels
codes = pd.Categorical(data.iloc[:, -1]).codes
data.iloc[:, -1] = codes

data.iloc[:, 0:-1] = data.iloc[:, 0:-1].astype('float32')
data.iloc[:,-1:  ] = data.iloc[:,-1:  ].astype('int32')

print(data.head())
print("Size:",  data.shape)

TRAIN_SIZE = 600
VAL_SIZE   = 200
TEST_SIZE  = 684

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

data_train.to_pickle("../data/yeast-train")
data_val.to_pickle("../data/yeast-val")
data_test.to_pickle("../data/yeast-test")

#--- prepare meta
idx = data.columns[:-1]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = data_train.mean()
meta['std'] = data_train.std()
meta['cost'] = np.random.randint(1, 5, size=(len(idx), 1)) / 5.
# meta['cost'] = 1.

meta = meta.astype('float32')

print("\nMeta:")
print(meta)

meta.to_pickle("../data/yeast-meta")