import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--data_folder", type=str, default='/home/hk303/xz/data/')
parser.add_argument("--optical_name", type=str, default='optical_test')
args = parser.parse_args()

optical_load = scipy.io.loadmat(args.data_folder + str(args.optical_name) + '.mat')

optical_key = list(optical_load.keys())
print(optical_key)

old_feature = optical_load['optical'].astype(float)

print(old_feature[0,0])
print(old_feature.dtype)
print(old_feature.shape)

pool_size = 4
h_size = old_feature.shape[0] // pool_size
v_size = old_feature.shape[1] // pool_size
new_feature = np.empty((h_size, v_size, old_feature.shape[2]))
print(new_feature.shape)
print(new_feature.dtype)
input()
for j in tqdm(range(h_size)):
    for k in range(v_size):
        for c in range(new_feature.shape[2]):
            new_feature[j, k, c] = np.mean(old_feature[j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size, c])   # 特征取均值
print(new_feature[0,0])

scipy.io.savemat('./data/optical_test_pool.mat', mdict={"feature": new_feature})
