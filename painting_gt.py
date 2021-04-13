import scipy.io
import numpy as np
import argparse
from scipy.stats import mode
from tqdm import tqdm
from PIL import Image

def pooling_data(old_labels, pool_size):
    h_size = old_labels.shape[0] // pool_size
    v_size = old_labels.shape[1] // pool_size
    new_labels = np.empty((h_size, v_size))

    for j in tqdm(range(h_size)):
        for k in range(v_size):
            new_labels[j, k] = mode(old_labels[j*pool_size:(j+1)*pool_size, k*pool_size:(k+1)*pool_size].reshape(-1))[0][0]       # 标签取众数
    return new_labels

def reshape_data(gt, size):
    pad_size = size // 2
    gt = np.pad(gt, ((pad_size,pad_size)), 'edge')
    h_patches = np.arange(size//2, gt.shape[0]-(size//2))
    h_size = len(h_patches)
    v_patches = np.arange(size//2, gt.shape[1]-(size//2))
    v_size = len(v_patches)
    gts = np.zeros((h_size * v_size), dtype=int)
    i = 0
    for h in tqdm(h_patches):
        for v in v_patches:
            gts[i] = gt[h, v]
            i += 1
    return gts

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

rgb_colors = np.array([[255, 0, 0], [255, 255, 0], [64, 0, 0], [138,43,226], [0, 255, 0], [0, 64, 0], [0, 128, 255]])

pool_size = 4
train_label = scipy.io.loadmat(args.data_folder + args.train_label + '.mat')
test_label = scipy.io.loadmat(args.data_folder + args.test_label + '.mat')

train_label_1 = train_label['label_1'] - 1  # (1200, 900)
train_label_2 = train_label['label_2'] - 1
train_label_3 = train_label['label_3'] - 1

test_label_1 = test_label['label_1'] - 1    # (300, 300)
test_label_2 = test_label['label_2'] - 1
test_label_3 = test_label['label_3'] - 1    # (150, 150)
test_label_4 = test_label['label_4'] - 1
test_label_5 = test_label['label_5'] - 1
train_labels_1 = pooling_data(train_label_1, pool_size=pool_size)
train_labels_2 = pooling_data(train_label_2, pool_size=pool_size)
train_labels_3 = pooling_data(train_label_3, pool_size=pool_size)
print('train_1:', train_labels_1.shape)
print('train_2:', train_labels_2.shape)
print('train_3:', train_labels_3.shape)
train_1 = reshape_data(train_labels_1, size=9)
train_2 = reshape_data(train_labels_2, size=9)
train_3 = reshape_data(train_labels_3, size=9)
print('train_1:', train_1.shape)
print('train_2:', train_2.shape)
print('train_3:', train_3.shape)

train_labels = {'train_1':train_1, 'train_2':train_2, 'train_3':train_3}
train_key = list(train_labels.keys())
train_val = list(train_labels.values())
h_img_train = 300
v_img_train = 225
for j in range(len(train_key)):
    img_out = Image.new("RGB", (h_img_train, v_img_train), "white")
    for i in tqdm(range(len(train_val[j]))):
        h_i = i // v_img_train
        v_i = i % v_img_train
        img_out.putpixel([h_i, v_i], (rgb_colors[train_val[j][i]][0], rgb_colors[train_val[j][i]][1], rgb_colors[train_val[j][i]][2]))
    img_out.save("./result_gt/%s.png"%(train_key[j]))


test_labels_1 = pooling_data(test_label_1, pool_size=pool_size)
test_labels_2 = pooling_data(test_label_2, pool_size=pool_size)
test_labels_3 = pooling_data(test_label_3, pool_size=pool_size)
test_labels_4 = pooling_data(test_label_4, pool_size=pool_size)
test_labels_5 = pooling_data(test_label_5, pool_size=pool_size)
print('test_1:', test_labels_1.shape)
print('test_2:', test_labels_2.shape)
print('test_3:', test_labels_3.shape)
print('test_4:', test_labels_4.shape)
print('test_5:', test_labels_5.shape)
test_1 = reshape_data(test_labels_1, size=9)
test_2 = reshape_data(test_labels_2, size=9)
test_3 = reshape_data(test_labels_3, size=9)
test_4 = reshape_data(test_labels_4, size=9)
test_5 = reshape_data(test_labels_5, size=9)
print('test_1:', test_1.shape)
print('test_2:', test_2.shape)
print('test_3:', test_3.shape)
print('test_4:', test_4.shape)
print('test_5:', test_5.shape)
test_labels_1 = {'test_1':test_1, 'test_2':test_2}
test_key_1 = list(test_labels_1.keys())
test_val_1 = list(test_labels_1.values())
h_img_test_1 = 75
v_img_test_1 = 75
for j in range(len(test_key_1)):
    img_out = Image.new("RGB", (h_img_test_1, v_img_test_1), "white")
    for i in tqdm(range(len(test_val_1[j]))):
        h_i = i // v_img_test_1
        v_i = i % v_img_test_1
        img_out.putpixel([h_i, v_i], (rgb_colors[test_val_1[j][i]][0], rgb_colors[test_val_1[j][i]][1], rgb_colors[test_val_1[j][i]][2]))
    img_out.save("./result_gt/%s.png"%(test_key_1[j]))

test_labels_2 = {'test_3':test_3, 'test_4':test_4, 'test_5':test_5}
test_key_2 = list(test_labels_2.keys())
test_val_2 = list(test_labels_2.values())
h_img_test_2 = 37
v_img_test_2 = 37
for j in range(len(test_key_2)):
    img_out = Image.new("RGB", (h_img_test_2, v_img_test_2), "white")
    for i in tqdm(range(len(test_val_2[j]))):
        h_i = i // v_img_test_2
        v_i = i % v_img_test_2
        img_out.putpixel([h_i, v_i], (rgb_colors[test_val_2[j][i]][0], rgb_colors[test_val_2[j][i]][1], rgb_colors[test_val_2[j][i]][2]))
    img_out.save("./result_gt/%s.png"%(test_key_2[j]))