import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm

def normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    data = (data - minValue) / (maxValue - minValue)
    return data

def reshape_data(stack, gt, size):
    pad_size = size // 2
    stack = np.pad(stack, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'edge')
    gt = np.pad(gt, ((pad_size,pad_size)), 'edge')
    h_patches = np.arange(size//2, stack.shape[0]-(size//2))
    h_size = len(h_patches)
    v_patches = np.arange(size//2, stack.shape[1]-(size//2))
    v_size = len(v_patches)

    stacks = np.zeros((h_size * v_size, stack.shape[2], size , size))
    gts = np.zeros((h_size * v_size), dtype=int)

    i = 0
    for h in tqdm(h_patches):
        for v in v_patches:
            for layer in range (stack.shape[2]):
                stacks[i][layer] = stack[(h-size//2):(h+size//2+1), (v-size//2):(v+size//2+1), layer]
            gts[i] = gt[h, v]
            i += 1
    return stacks, gts

def reshape_nolabel_data(stack, size):
    pad_size = size // 2
    stack = np.pad(stack, ((pad_size,pad_size),(pad_size,pad_size),(0,0)), 'edge')
    h_patches = np.arange(size//2, stack.shape[0]-(size//2))
    h_size = len(h_patches)
    v_patches = np.arange(size//2, stack.shape[1]-(size//2))
    v_size = len(v_patches)

    stacks = np.zeros((h_size * v_size, stack.shape[2], size , size))

    i = 0
    for h in tqdm(h_patches):
        for v in v_patches:
            for layer in range (stack.shape[2]):
                stacks[i][layer] = stack[(h-size//2):(h+size//2+1), (v-size//2):(v+size//2+1), layer]
            i += 1
    
    return stacks

def pooling_data(old_feature, old_labels, pool_size):
    h_size = old_feature.shape[1] // pool_size
    v_size = old_feature.shape[2] // pool_size
    new_feature = np.empty((old_feature.shape[0], h_size, v_size, old_feature.shape[3]))
    new_labels = np.empty((old_feature.shape[0], h_size, v_size))

    for i in range(old_feature.shape[0]):
        for j in tqdm(range(h_size)):
            for k in range(v_size):
                for c in range(new_feature.shape[3]):
                    new_feature[i][j, k, c] = np.mean(old_feature[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size, c])   # 特征取均值
                new_labels[i][j, k] = mode(old_labels[i][j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size].reshape(-1))[0][0]       # 标签取众数
    return new_feature, new_labels

def pooling_nolabel_data(old_feature, pool_size):
    h_size = old_feature.shape[0] // pool_size
    v_size = old_feature.shape[1] // pool_size
    new_feature = np.empty((h_size, v_size, old_feature.shape[2]))

    for j in tqdm(range(h_size)):
        for k in range(v_size):
            for c in range(new_feature.shape[2]):
                new_feature[j, k, c] = np.mean(old_feature[j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size, c])   # 特征取均值
    return new_feature

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--data_folder", type=str, default='./data/')
parser.add_argument("--optical_train", type=str, default='optical_train')
parser.add_argument("--optical_test", type=str, default='test_optical')
parser.add_argument("--optical_valid", type=str, default='optical_test')
parser.add_argument("--sar_train", type=str, default='sar_train')
parser.add_argument("--sar_test", type=str, default='test_sar')
parser.add_argument("--sar_valid", type=str, default='sar_test')
parser.add_argument("--train_label", type=str, default='train_label')
parser.add_argument("--test_label", type=str, default='test_label')
args = parser.parse_args()
#--------------------------------------------------------------------------------#
pool_size = 4
optical_train = scipy.io.loadmat(args.data_folder + args.optical_train + '.mat')
optical_test = scipy.io.loadmat(args.data_folder + args.optical_test + '.mat')
optical_valid = scipy.io.loadmat(args.data_folder + args.optical_valid + '.mat')

sar_train = scipy.io.loadmat(args.data_folder + args.sar_train + '.mat')
sar_test = scipy.io.loadmat(args.data_folder + args.sar_test + '.mat')
sar_valid = scipy.io.loadmat(args.data_folder + args.sar_valid + '.mat')

train_label = scipy.io.loadmat(args.data_folder + args.train_label + '.mat')
test_label = scipy.io.loadmat(args.data_folder + args.test_label + '.mat')
#--------------------------------------------------------------------------------#
optical_train_key = list(optical_train.keys())
optical_test_key = list(optical_test.keys())
optical_valid_key = list(optical_valid.keys())
sar_train_key = list(sar_train.keys())
sar_test_key = list(sar_test.keys())
sar_valid_key = list(sar_valid.keys())
train_label_key = list(train_label.keys())
test_label_key = list(test_label.keys())

# print(optical_train_key, optical_test_key, optical_valid_key, sar_train_key, sar_test_key, sar_valid_key, train_label_key, test_label_key)
#--------------------------------------------------------------------------------#
optical_train_1 = optical_train['optical_1']
optical_train_2 = optical_train['optical_2']
optical_train_3 = optical_train['optical_3']

optical_test_1 = optical_test['optical_1']
optical_test_2 = optical_test['optical_2']
optical_test_3 = optical_test['optical_3']
optical_test_4 = optical_test['optical_4']
optical_test_5 = optical_test['optical_5']

optical_valid = optical_valid['optical'].astype(float)
print("valid: ", optical_valid.shape)
sar_train_1 = np.expand_dims(sar_train['sar_1'], 2)
sar_train_2 = np.expand_dims(sar_train['sar_2'], 2)
sar_train_3 = np.expand_dims(sar_train['sar_3'], 2)

sar_test_1 = np.expand_dims(sar_test['sar_1'], 2)
sar_test_2 = np.expand_dims(sar_test['sar_2'], 2)
sar_test_3 = np.expand_dims(sar_test['sar_3'], 2)
sar_test_4 = np.expand_dims(sar_test['sar_4'], 2)
sar_test_5 = np.expand_dims(sar_test['sar_5'], 2)

sar_valid = np.expand_dims(sar_valid['sar'].astype(float), 2)

train_label_1 = train_label['label_1']
train_label_2 = train_label['label_2']
train_label_3 = train_label['label_3']

test_label_1 = test_label['label_1']
test_label_2 = test_label['label_2']
test_label_3 = test_label['label_3']
test_label_4 = test_label['label_4']
test_label_5 = test_label['label_5']
#--------------------------------------------------------------------------------#
optical_train = np.asarray([optical_train_1, optical_train_2, optical_train_3], dtype=float)
optical_test_300 = np.asarray([optical_test_1, optical_test_2], dtype=float)
optical_test_150 = np.asarray([optical_test_3, optical_test_4, optical_test_5], dtype=float)

sar_train = np.asarray([sar_train_1, sar_train_2, sar_train_3], dtype=float)
sar_test_300 = np.asarray([sar_test_1, sar_test_2], dtype=float)
sar_test_150 = np.asarray([sar_test_3, sar_test_4, sar_test_5], dtype=float)

train_labels = np.asarray([train_label_1, train_label_2, train_label_3])
test_labels_300 = np.asarray([test_label_1, test_label_2])
test_labels_150 = np.asarray([test_label_3, test_label_4, test_label_5])
#--------------------------------------------------------------------------------#
optical_train_new, train_labels_new = pooling_data(optical_train, train_labels, pool_size=pool_size)
optical_train_new[0] = normalize(optical_train_new[0])
optical_train_stacks, optical_train_gts = reshape_data(optical_train_new[0], train_labels_new[0], size=9)
for i in range(1, optical_train_new.shape[0]):
    optical_train_new[i] = normalize(optical_train_new[i])
    optical_train_stacks_temp, optical_train_gts_temp = reshape_data(optical_train_new[i],train_labels_new[i], size=9)
    optical_train_stacks = np.concatenate((optical_train_stacks, optical_train_stacks_temp), axis=0)
    optical_train_gts = np.concatenate((optical_train_gts, optical_train_gts_temp), axis=0)
# print(optical_train_stacks.shape, optical_train_gts.shape)    #(202500,1,9,9)
#--------------------------------------------------------------------------------#
sar_train_new, train_labels_new = pooling_data(sar_train, train_labels, pool_size=pool_size)
sar_train_new[0] = normalize(sar_train_new[0])
sar_train_stacks, sar_train_gts = reshape_data(sar_train_new[0], train_labels_new[0], size=9)
for i in range(1, sar_train_new.shape[0]):
    sar_train_new[i] = normalize(sar_train_new[i])
    sar_train_stacks_temp, sar_train_gts_temp = reshape_data(sar_train_new[i],train_labels_new[i], size=9)
    sar_train_stacks = np.concatenate((sar_train_stacks, sar_train_stacks_temp), axis=0)
    sar_train_gts = np.concatenate((sar_train_gts, sar_train_gts_temp), axis=0)
# print(sar_train_stacks.shape, sar_train_gts.shape)    #(202500,1,9,9)
#--------------------------------------------------------------------------------#
optical_test_new_300, test_labels_new_300 = pooling_data(optical_test_300, test_labels_300, pool_size=pool_size)
optical_test_new_300[0] = normalize(optical_test_new_300[0])
optical_test_stacks_300, optical_test_gts_300 = reshape_data(optical_test_new_300[0], test_labels_new_300[0], size=9)
for i in range(1, optical_test_new_300.shape[0]):
    optical_test_new_300[i] = normalize(optical_test_new_300[i])
    optical_test_stacks_temp_300, optical_test_gts_temp_300 = reshape_data(optical_test_new_300[i],test_labels_new_300[i], size=9)
    optical_test_stacks_300 = np.concatenate((optical_test_stacks_300, optical_test_stacks_temp_300), axis=0)
    optical_test_gts_300 = np.concatenate((optical_test_gts_300, optical_test_gts_temp_300), axis=0)

optical_test_new_150, test_labels_new_150 = pooling_data(optical_test_150, test_labels_150, pool_size=pool_size)
optical_test_new_150[0] = normalize(optical_test_new_150[0])
optical_test_stacks_150, optical_test_gts_150 = reshape_data(optical_test_new_150[0], test_labels_new_150[0], size=9)
for i in range(1, optical_test_new_150.shape[0]):
    optical_test_new_150[i] = normalize(optical_test_new_150[i])
    optical_test_stacks_temp_150, optical_test_gts_temp_150 = reshape_data(optical_test_new_150[i],test_labels_new_150[i], size=9)
    optical_test_stacks_150 = np.concatenate((optical_test_stacks_150, optical_test_stacks_temp_150), axis=0)
    optical_test_gts_150 = np.concatenate((optical_test_gts_150, optical_test_gts_temp_150), axis=0)

optical_test_stacks = np.concatenate((optical_test_stacks_300, optical_test_stacks_150), axis=0)
optical_test_gts = np.concatenate((optical_test_gts_300, optical_test_gts_150), axis=0)
# print(optical_test_stacks.shape, optical_test_gts.shape)    #(15357,1,9,9)
#--------------------------------------------------------------------------------#
sar_test_new_300, test_labels_new_300 = pooling_data(sar_test_300, test_labels_300, pool_size=pool_size)

sar_test_new_300[0] = normalize(sar_test_new_300[0])
sar_test_stacks_300, sar_test_gts_300 = reshape_data(sar_test_new_300[0], test_labels_new_300[0], size=9)
for i in range(1, sar_test_new_300.shape[0]):
    sar_test_new_300[i] = normalize(sar_test_new_300[i])
    sar_test_stacks_temp_300, sar_test_gts_temp_300 = reshape_data(sar_test_new_300[i],test_labels_new_300[i], size=9)
    sar_test_stacks_300 = np.concatenate((sar_test_stacks_300, sar_test_stacks_temp_300), axis=0)
    sar_test_gts_300 = np.concatenate((sar_test_gts_300, sar_test_gts_temp_300), axis=0)

sar_test_new_150, test_labels_new_150 = pooling_data(sar_test_150, test_labels_150, pool_size=pool_size)

sar_test_new_150[0] = normalize(sar_test_new_150[0])
sar_test_stacks_150, sar_test_gts_150 = reshape_data(sar_test_new_150[0], test_labels_new_150[0], size=9)
for i in range(1, sar_test_new_150.shape[0]):
    sar_test_new_150[i] = normalize(sar_test_new_150[i])
    sar_test_stacks_temp_150, sar_test_gts_temp_150 = reshape_data(sar_test_new_150[i],test_labels_new_150[i], size=9)
    sar_test_stacks_150 = np.concatenate((sar_test_stacks_150, sar_test_stacks_temp_150), axis=0)
    sar_test_gts_150 = np.concatenate((sar_test_gts_150, sar_test_gts_temp_150), axis=0)

sar_test_stacks = np.concatenate((sar_test_stacks_300, sar_test_stacks_150), axis=0)
sar_test_gts = np.concatenate((sar_test_gts_300, sar_test_gts_150), axis=0)
# print(sar_test_stacks.shape, sar_test_gts.shape)    #(15357,1,9,9)
#--------------------------------------------------------------------------------#
# print(optical_test_gts==sar_test_gts)
# print(optical_train_gts==sar_train_gts)
#--------------------------------------------------------------------------------#
optical_valid_new = pooling_nolabel_data(optical_valid, pool_size=pool_size)
print("valid_new: ", optical_valid_new.shape)   # (1500, 1000, 3)
optical_valid_new = normalize(optical_valid_new)
optical_valid_stacks = reshape_nolabel_data(optical_valid_new, size=9)
print("optical_valid_stacks: ", optical_valid_stacks.shape)
input("gg")
# print(optical_valid_stacks.shape)   #(1500000,1,9,9)
#--------------------------------------------------------------------------------#
sar_valid_new = pooling_nolabel_data(sar_valid, pool_size=pool_size)
sar_valid_new = normalize(sar_valid_new)
sar_valid_stacks = reshape_nolabel_data(sar_valid_new, size=9)
# print(sar_valid_stacks.shape)   #(1500000,1,9,9)
#--------------------------------------------------------------------------------#
scipy.io.savemat('./pre_data/optical_train.mat', mdict={"optical_train": optical_train_stacks})
scipy.io.savemat('./pre_data/optical_test.mat', mdict={"optical_test": optical_test_stacks})


scipy.io.savemat('./pre_data/sar_train.mat', mdict={"sar_train": sar_train_stacks})
scipy.io.savemat('./pre_data/sar_test.mat', mdict={"sar_test": sar_test_stacks})


scipy.io.savemat('./pre_data/train_label.mat', mdict={"train_label": optical_train_gts})
scipy.io.savemat('./pre_data/test_label.mat', mdict={"test_label": optical_test_gts})

scipy.io.savemat('./pre_data/optical_valid.mat', mdict={"optical_valid": optical_valid_stacks})
scipy.io.savemat('./pre_data/sar_valid.mat', mdict={"sar_valid": sar_valid_stacks})
