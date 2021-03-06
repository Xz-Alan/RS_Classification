import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--data_folder", type=str, default='../data/')
parser.add_argument("--optical_name", type=str, default='optical_train')
parser.add_argument("--sar_name", type=str, default='sar_train')
parser.add_argument("--grey_name", type=str, default='grey_train')
parser.add_argument("--labels_name", type=str, default='train_label')
args = parser.parse_args()

optical_load = scipy.io.loadmat(args.data_folder + str(args.optical_name) + '.mat')
sar_load = scipy.io.loadmat(args.data_folder + str(args.sar_name) + '.mat')
grey_load = scipy.io.loadmat(args.data_folder + str(args.grey_name) + '.mat')
label_load = scipy.io.loadmat(args.data_folder + str(args.labels_name) + '.mat')

optical_key = list(optical_load.keys())
sar_key = list(sar_load.keys())
grey_key = list(grey_load.keys())
label_key = list(label_load.keys())
'''
print(optical_key)
print("-"*100)
print(sar_key)
print("-"*100)
print(grey_key)
print("-"*100)
print(label_key)
input()
'''
optical_1 = optical_load['optical_1']; labels_1 = label_load['label_1']
optical_2 = optical_load['optical_2']; labels_2 = label_load['label_2']
optical_3 = optical_load['optical_3']; labels_3 = label_load['label_3']
sar_1 = np.expand_dims(sar_load['sar_1'], 2); grey_1 = np.expand_dims(grey_load['grey_1'], 2)
sar_2 = np.expand_dims(sar_load['sar_2'], 2); grey_2 = np.expand_dims(grey_load['grey_2'], 2)
sar_3 = np.expand_dims(sar_load['sar_3'], 2); grey_3 = np.expand_dims(grey_load['grey_3'], 2)

print(optical_1[0,0])
print(sar_1[0,0])
print(grey_1[0,0])
print(labels_1[0,0])
input()

feature_1 = np.concatenate((optical_1,sar_1,grey_1),axis=2)
feature_2 = np.concatenate((optical_2,sar_2,grey_2),axis=2)
feature_3 = np.concatenate((optical_3,sar_3,grey_3),axis=2)
print(feature_1.shape)

old_feature = np.asarray([feature_1, feature_2, feature_3], dtype=float)    # (3,1200,900)
old_labels = np.asarray([labels_1, labels_2, labels_3])   # (3,1200,900)
# scipy.io.savemat('./data/sar_train_nopool.mat', mdict={"feature": old_feature,"labels": old_labels})
print(old_feature.shape)
print(old_labels.shape)
pool_size = 2
h_size = feature_1.shape[0] // pool_size
v_size = feature_1.shape[1] // pool_size
new_feature = np.empty((old_feature.shape[0], h_size, v_size, feature_1.shape[2]))  # (3,240,180)
new_labels = np.empty((old_feature.shape[0], h_size, v_size))   # (3,240,180)
print(new_feature.shape)
print(new_labels.shape)

for i in range(new_feature.shape[0]):
    for j in tqdm(range(h_size)):
        for k in range(v_size):
            for c in range(new_feature.shape[3]):
                new_feature[i][j, k, c] = np.mean(old_feature[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size, c])   # ???????????????

            new_labels[i][j, k] = mode(old_labels[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size].reshape(-1))[0][0]       # ???????????????
print(new_feature[1].dtype)
print(new_labels[0])

scipy.io.savemat('./data/fusion_train_pool.mat', mdict={"feature": new_feature,"labels": new_labels})
