# Instructions:
# =============
# 1. Open cifar10_cs.mat file in octave (available upon request)
# 2. Extract csv files:

'''
# convert in octave:

csvwrite("cifar-train-x.csv", xtr)
csvwrite("cifar-train-y.csv", ytr)

csvwrite("cifar-val-x.csv", xtv)
csvwrite("cifar-val-y.csv", ytv)

csvwrite("cifar-test-x.csv", xte)
csvwrite("cifar-test-y.csv", yte)
'''

# 3. Run this script

import pandas as pd
import numpy as np

COL_LABEL = '_label'

SEED = 998823
#---
np.random.seed(SEED)

data_train_x = pd.read_csv("../data/raw/cifar-train-x.csv", header=None, sep=',')
data_train_y = pd.read_csv("../data/raw/cifar-train-y.csv", header=None, sep=',')
data_val_x = pd.read_csv("../data/raw/cifar-val-x.csv", header=None, sep=',')
data_val_y = pd.read_csv("../data/raw/cifar-val-y.csv", header=None, sep=',')
data_test_x = pd.read_csv("../data/raw/cifar-test-x.csv", header=None, sep=',')
data_test_y = pd.read_csv("../data/raw/cifar-test-y.csv", header=None, sep=',')

print("Loaded")

def conv(x, y):
	y[y==-1] = 0

	data = x.transpose().fillna(value=0).astype('float32')
	data[COL_LABEL] = y.transpose().astype('int32')

	return data

data_train = conv(data_train_x, data_train_y)	
data_val   = conv(data_val_x,   data_val_y)	
data_test  = conv(data_test_x,  data_test_y)	

print(data_test.head())
print()
print("Train len:", data_train.shape[0])
print("Val len:  ", data_val.shape[0])
print("Test len: ", data_test.shape[0])

data_train.to_pickle("../data/cifar-train")
data_val.to_pickle("../data/cifar-val")
data_test.to_pickle("../data/cifar-test")

#--- prepare meta
idx = data_train.columns[:-1]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = 0.
meta['std'] = 1.
# meta['_avg'] = data_train.mean()
# meta['_std'] = data_train.std()
meta['absmax'] = data_train.abs().max()
meta['cost'] = 1.

meta = meta.astype('float32')

print(meta)
meta.to_pickle("../data/cifar-meta")
