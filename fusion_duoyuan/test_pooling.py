import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--data_folder", type=str, default='/home/hk303/xz/data/')
parser.add_argument("--optical_name", type=str, default='optical_test')
parser.add_argument("--sar_name", type=str, default='sar_test')
parser.add_argument("--grey_name", type=str, default='grey_test')
args = parser.parse_args()

optical_load = scipy.io.loadmat(args.data_folder + str(args.optical_name) + '.mat')
sar_load = scipy.io.loadmat(args.data_folder + str(args.sar_name) + '.mat')
grey_load = scipy.io.loadmat(args.data_folder + str(args.grey_name) + '.mat')

optical_key = list(optical_load.keys())
sar_key = list(sar_load.keys())
grey_key = list(grey_load.keys())

print(optical_key)
print("-"*100)
print(sar_key)
print("-"*100)
print(grey_key)


optical = optical_load['optical']
sar = np.expand_dims(sar_load['sar'], 2)
grey = np.expand_dims(grey_load['grey'], 2)

print(optical[0,0])
print(sar[0,0])
print(grey[0,0])


old_feature = np.concatenate((optical, sar, grey), axis=2).astype(float)
print(old_feature[0,0])
print(old_feature.dtype)

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

scipy.io.savemat('./data/fusion_test_pool.mat', mdict={"feature": new_feature})
