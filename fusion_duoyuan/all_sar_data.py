import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import random_split, TensorDataset, DataLoader
from tqdm import tqdm

def normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    data = (data - minValue) / (maxValue - minValue)
    return data

def mat_data(args):
    data_load = scipy.io.loadmat(args.data_folder + str(args.que_data_name) + '.mat')
    key = list(data_load.keys())
    # print(key)
    # input()
    # 特征导入
    stack = data_load['feature'].astype(float)    # (1500,1000,3)
    print(stack.shape)
    stack = np.pad(stack, ((8,8),(8,8),(0,0)), 'edge')
    print(stack.shape)
    print("stack: ", stack.dtype)
    return stack


def reshape_rectangle_data(stack, size_1, size_2, size_3):
    h_patches = np.arange(size_3//2, stack.shape[0]-(size_3//2))
    h_size = len(h_patches)
    v_patches = np.arange(size_3//2, stack.shape[1]-(size_3//2))
    v_size = len(v_patches)

    stacks_1 = np.zeros((h_size * v_size, stack.shape[2], size_1 , size_1))
    stacks_2 = np.zeros((h_size * v_size, stack.shape[2], size_2 , size_2))
    stacks_3 = np.zeros((h_size * v_size, stack.shape[2], size_3 , size_3))

    i = 0
    for h in tqdm(h_patches):
        for v in v_patches:
            for layer in range (stack.shape[2]):
                stacks_1[i][layer] = stack[(h-size_1//2):(h+size_1//2+1), (v-size_1//2):(v+size_1//2+1), layer]
                stacks_2[i][layer] = stack[(h-size_2//2):(h+size_2//2+1), (v-size_2//2):(v+size_2//2+1), layer]
                stacks_3[i][layer] = stack[(h-size_3//2):(h+size_3//2+1), (v-size_3//2):(v+size_3//2+1), layer]
            i += 1
    '''
    print(f"stacks_1: {stacks_1.shape}")    # (739872,27,5,5)
    print(f"stacks_2: {stacks_2.shape}")    # (739872,27,11,11)
    print(f"stacks_3: {stacks_3.shape}")    # (739872,27,17,17)
    print(f"gts: {gts.shape}")      # (750360)
    '''
    return stacks_1, stacks_2, stacks_3

def sar_datesets(args):
    stack= mat_data(args)
    stack = normalize(stack)  # 特征归一化
    stacks_1, stacks_2, stacks_3 = reshape_rectangle_data(stack, args.sar_size1, args.sar_size2, args.sar_size3)
    
    print(f"Resizing image of size {stack.shape} to image patches {stacks_1.shape}, {stacks_2.shape} and {stacks_3.shape}")
    np.save('./data/' + args.que_data_name + '/stacks_1.npy', stacks_1)
    np.save('./data/' + args.que_data_name + '/stacks_2.npy', stacks_2)
    np.save('./data/' + args.que_data_name + '/stacks_3.npy', stacks_3)
    print("+++++++++")
    return 1

def sar_dataloader(args, gts_class, gts, stacks_1, stacks_2, stacks_3, split='train', form='support', shuffle=True):
    # init parameters
    if split == 'train':
        if form == 'support':
            n_shot = args.train_n_shot
        elif form == 'query':
            n_shot = args.train_n_query
        else:
            print("form error")
    elif split == 'test':
        if form == 'support':
            n_shot = args.test_n_shot
        elif form == 'query':
            n_shot = args.test_n_query
        else:
            print("form error")
    else:
        print("split error")
    stack_index = np.arange(0, gts.size(0))   # 生成stack的索引
    index = np.zeros((1,  2), dtype=int)    # 生成一个零数组，方便for循环
    class_num = np.zeros(args.test_n_way).astype(int)
    j = 0
    for i in gts_class:
        stack_index_i = stack_index[gts == i]
        gts_index_i = np.ones(n_shot, dtype=int)*j
        gts_index_i = gts_index_i[:, np.newaxis]    # 增加维度
        class_num[i] = len(stack_index_i)
        # print(i, ":", len(stack_index_i))
        stack_index_i = np.random.choice(stack_index_i, n_shot, False)
        # print("stack_index_i: ", stack_index_i)
        stack_index_i = stack_index_i[:, np.newaxis]
        index_i = np.concatenate((stack_index_i, gts_index_i), axis=1)
        index = np.concatenate((index, index_i), axis=0)
        j += 1
    
    if shuffle :
        index = np.random.permutation(np.delete(index, 0 , 0))  # 去除第一个值并打乱顺序
    else:
        index = np.delete(index, 0 , 0)     # 不打乱顺序
    # print("index: ", index)
    # print("gts: ", gts[133829], gts[181901], gts[21650], gts[51858])
    epoch_stacks_1 = []
    epoch_stacks_2 = []
    epoch_stacks_3 = []
    epoch_gts = torch.from_numpy(index[:,1])
    for item in list(index[:,0]):
        epoch_stacks_1.append(stacks_1[item].unsqueeze(0))  # 每一行需要增加一维，拼接时保证维度正确
        epoch_stacks_2.append(stacks_2[item].unsqueeze(0))
        epoch_stacks_3.append(stacks_3[item].unsqueeze(0))
    epoch_stacks_1 = torch.cat(epoch_stacks_1, dim=0)   # (25,27,5,5)
    epoch_stacks_2 = torch.cat(epoch_stacks_2, dim=0)
    epoch_stacks_3 = torch.cat(epoch_stacks_3, dim=0)
    return epoch_stacks_1, epoch_stacks_2, epoch_stacks_3, epoch_gts, class_num
