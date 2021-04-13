import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

parser = argparse.ArgumentParser(description="pooling")
parser.add_argument("--data_folder", type=str, default='/home/hk303/xz/data/')
parser.add_argument("--data_name", type=str, default='grey_test')
args = parser.parse_args()

data_load = scipy.io.loadmat(args.data_folder + str(args.data_name) + '.mat')
data_key = list(data_load.keys())
print(data_key)

old_feature = data_load['grey'].astype(float)
print(old_feature.shape)
print(old_feature.dtype)

pool_size = 4
h_size = old_feature.shape[0] // pool_size
v_size = old_feature.shape[1] // pool_size
new_feature = np.empty((h_size, v_size))  # (3,240,180)
print(new_feature.shape)
print(new_feature.dtype)
input()
for j in tqdm(range(h_size)):
    for k in range(v_size):
        new_feature[j, k] = np.mean(old_feature[j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size])   # 特征取均值

scipy.io.savemat('./data/grey_test_pool.mat', mdict={"feature": new_feature})
