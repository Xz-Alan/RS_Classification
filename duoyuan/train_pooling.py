import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--data_folder", type=str, default='../data/')
parser.add_argument("--data_name", type=str, default='sar_train')
parser.add_argument("--labels_name", type=str, default='train_label')
args = parser.parse_args()

data_load = scipy.io.loadmat(args.data_folder + str(args.data_name) + '.mat')
label_load = scipy.io.loadmat(args.data_folder + str(args.labels_name) + '.mat')
data_key = list(data_load.keys())
label_key = list(label_load.keys())
print(data_key)
print("*"*100)
print(label_key)
input()
feature_1 = data_load['sar_1']; labels_1 = label_load['label_1']
feature_2 = data_load['sar_2']; labels_2 = label_load['label_2']
feature_3 = data_load['sar_3']; labels_3 = label_load['label_3']

print(feature_1.shape)
print(labels_1.shape)
old_feature = np.asarray([feature_1, feature_2, feature_3], dtype=float)    # (3,1200,900)
old_labels = np.asarray([labels_1, labels_2, labels_3])   # (3,1200,900)
# scipy.io.savemat('./data/sar_train_nopool.mat', mdict={"feature": old_feature,"labels": old_labels})

pool_size = 2
h_size = feature_1.shape[0] // pool_size
v_size = feature_1.shape[1] // pool_size
new_feature = np.empty((old_feature.shape[0], h_size, v_size))  # (3,240,180)
new_labels = np.empty((old_feature.shape[0], h_size, v_size))   # (3,240,180)
print(new_feature.shape, new_labels.shape)

for i in range(new_feature.shape[0]):
    for j in tqdm(range(h_size)):
        for k in range(v_size):
            new_feature[i][j, k] = np.mean(old_feature[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size])   # 特征取均值

            new_labels[i][j, k] = mode(old_labels[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size].reshape(-1))[0][0]       # 标签取众数
print(new_feature[1].dtype)
print(new_labels[0])

scipy.io.savemat('./data/sar_train_pool.mat', mdict={"feature": new_feature,"labels": new_labels})
