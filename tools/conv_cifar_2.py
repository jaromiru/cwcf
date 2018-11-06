from torchvision import datasets, transforms
import pandas as pd
import numpy as np

COLUMN_LABEL = '_label'
SEED = 998822
#---
np.random.seed(SEED)

#---
def get_data(train):
	data_raw = datasets.CIFAR10('../data/dl/', train=train, download=True,  transform=transforms.Compose([
							transforms.Grayscale(),
							transforms.Resize((20, 20)),
							transforms.ToTensor(),
							lambda x: x.numpy().flatten()]))

	data_x, data_y = zip(*data_raw)
	
	data_x = np.array(data_x)
	data_y = np.array(data_y, dtype='int32').reshape(-1, 1)

	# binarize
	label_0 = data_y < 5
	label_1 = ~label_0

	data_y[label_0] = 0
	data_y[label_1] = 1

	data = pd.DataFrame(data_x)
	data[COLUMN_LABEL] = data_y

	return data, data_x.mean(), data_x.std()

#---
data_train, avg, std = get_data(train=True)
data_test, _, _  = get_data(train=False)

# shuffle
val_idx = np.random.choice(data_train.shape[0], 10000, replace=False).tolist()

data_val   = data_train.iloc[val_idx]
data_train = data_train.drop(val_idx)

print(data_train.head())

print("Number of features:", data_train.shape[1] - 1)
print("Classes:", data_train.iloc[:, -1].unique())

print()
print("Train len:", data_train.shape[0])
print("Val len:  ", data_val.shape[0])
print("Test len: ", data_test.shape[0])

data_train.to_pickle("../data/cifar-2-train")
data_val.to_pickle("../data/cifar-2-val")
data_test.to_pickle("../data/cifar-2-test")

#--- prepare meta
idx = data_train.columns[:-1]
meta = pd.DataFrame(index=idx, dtype='float32')

meta['avg'] = avg	#data_train.mean()
meta['std'] = std	#data_train.std()
meta['cost'] = 1.

meta.loc[ meta['std'] == 0., 'std' ] = 1.0
meta = meta.astype('float32')

print()
print(meta)

meta.to_pickle("../data/cifar-2-meta")
