import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import random_split, TensorDataset, DataLoader

def normalize(data):
    maxValue = np.max(data)
    minValue = np.min(data)
    data = (data - minValue) / (maxValue - minValue)
    return data


def mat_data(args):
    data_load = scipy.io.loadmat(args.data_folder + str(args.data_name) + '.mat')
    # key = list(data_load.keys())
    # print(key)
    # 特征导入
    stack = data_load['feature']
    gt = data_load['labels']
    print(stack.shape, gt.shape)
    
    s = np.zeros(7, dtype=int)
    for n in range(gt.shape[0]):
        for i in range(gt.shape[1]):
            for j in range(gt.shape[2]):
                for k in range(7):
                    if gt[n][i][j] == k+1:
                        s[k] += 1
                        break
    for k in range(7):
        print(k+1,": ", s[k])
    
    print("num_class: ", np.max(gt))
    print("stack: ", stack.dtype)
    # input()
    return stack, gt

def reshape_data(stack,gt):
    patches_h = stack.shape[0]
    patches_v = stack.shape[1]
    stacks = torch.zeros(patches_h*patches_v, stack.shape[2])
    gts = torch.zeros(patches_h*patches_v).int()
    i = 0
    filled = 0
    for h in range(patches_h):
        for v in range(patches_v):
            for layer in range (stack.shape[2]):
                stacks[i][layer] = stack[h, v, layer]
            gts[i] = gt[h, v]
            i += 1
    return stacks, gts

def reshape_rectangle_data(stack, gt, size_1, size_2, size_3):
    h_patches = np.arange(size_3//2, stack.shape[0]-(size_3//2))
    h_size = len(h_patches)
    v_patches = np.arange(size_3//2, stack.shape[1]-(size_3//2))
    v_size = len(v_patches)

    stacks_1 = np.zeros((h_size * v_size, size_1 , size_1))
    stacks_2 = np.zeros((h_size * v_size, size_2 , size_2))
    stacks_3 = np.zeros((h_size * v_size, size_3 , size_3))
    gts = np.zeros((h_size * v_size), dtype=int)

    i = 0
    for h in h_patches:
        for v in v_patches:
            stacks_1[i] = stack[(h-size_1//2):(h+size_1//2+1), (v-size_1//2):(v+size_1//2+1)]
            stacks_2[i] = stack[(h-size_2//2):(h+size_2//2+1), (v-size_2//2):(v+size_2//2+1)]
            stacks_3[i] = stack[(h-size_3//2):(h+size_3//2+1), (v-size_3//2):(v+size_3//2+1)]
            gts[i] = gt[h, v]
            i += 1
    
    print(f"stacks_1: {stacks_1.shape}")    # (739872,3,5,5)
    print(f"stacks_2: {stacks_2.shape}")    # (739872,3,11,11)
    print(f"stacks_3: {stacks_3.shape}")    # (739872,3,17,17)
    print(f"gts: {gts.shape}")      # (750360)
    return stacks_1, stacks_2, stacks_3, gts

def sar_datesets(args):
    
    stack, gt = mat_data(args)
    stack[0] = normalize(stack[0])  # 特征归一化
    stacks_1, stacks_2, stacks_3, gts = reshape_rectangle_data(stack[0],gt[0],args.sar_size1,args.sar_size2,args.sar_size3)
    for i in range(1, stack.shape[0]):
        stack[i] = normalize(stack[i])  # 特征归一化
        stacks_1_temp, stacks_2_temp, stacks_3_temp, gts_temp = reshape_rectangle_data(stack[i],gt[i],args.sar_size1,args.sar_size2,args.sar_size3)
        stacks_1 = np.concatenate((stacks_1, stacks_1_temp), axis=0)
        stacks_2 = np.concatenate((stacks_2, stacks_2_temp), axis=0)
        stacks_3 = np.concatenate((stacks_3, stacks_3_temp), axis=0)
        gts = np.concatenate((gts, gts_temp), axis=0)
    print(stacks_1.shape, stacks_2.shape, stacks_3.shape, gts.shape)
    print(f"Resizing image of size {stack.shape} to image patches {stacks_1.shape}, {stacks_2.shape} and {stacks_3.shape}")
    np.save('./data/' + args.data_name + '/stacks_1.npy', stacks_1)
    np.save('./data/' + args.data_name + '/stacks_2.npy', stacks_2)
    np.save('./data/' + args.data_name + '/stacks_3.npy', stacks_3)
    np.save('./data/' + args.data_name + '/gts.npy', gts)
    input("+++++")
    
    stacks_1 = np.load('./data/' + args.data_name + '/stacks_1.npy')
    stacks_2 = np.load('./data/' + args.data_name + '/stacks_2.npy')
    stacks_3 = np.load('./data/' + args.data_name + '/stacks_3.npy')
    gts = np.load('./data/' + args.data_name + '/gts.npy')

    s = np.zeros(7, dtype=int)
    for n in range(gts.shape[0]):
        for k in range(7):
            if gts[n] == k+1:
                s[k] += 1
                break
    for k in range(7):
        print(k+1,": ", s[k])

    index = np.arange(gts.shape[0])
    index_i = index[gts == 7]
    train_index = index_i[:2000]
    test_index = index_i[1500:]
    for i in range(np.max(gts)):
        index_i = index[gts == i]
        if len(index_i) != 0:
            np.random.shuffle(index_i)  # 打乱顺序
            train_index = np.concatenate((train_index, index_i[:10000]), axis=0)
            test_index =np.concatenate((test_index, index_i[10000:]),axis=0)
    # train_index = np.delete(train_index, 0)
    # test_index = np.delete(test_index, 0)

    train_stacks_1 = stacks_1[train_index]
    train_stacks_2 = stacks_2[train_index]
    train_stacks_3 = stacks_3[train_index]
    train_gts = gts[train_index]
    
    test_stacks_1 = stacks_1[test_index]
    test_stacks_2 = stacks_2[test_index]
    test_stacks_3 = stacks_3[test_index]
    test_gts = gts[test_index]
    print(train_stacks_1.shape, test_stacks_1.shape)
    np.save('./data/' + args.data_name + '/train_stacks_1.npy', train_stacks_1)
    np.save('./data/' + args.data_name + '/train_stacks_2.npy', train_stacks_2)
    np.save('./data/' + args.data_name + '/train_stacks_3.npy', train_stacks_3)
    np.save('./data/' + args.data_name + '/train_gts.npy', train_gts)
    np.save('./data/' + args.data_name + '/test_stacks_1.npy', test_stacks_1)
    np.save('./data/' + args.data_name + '/test_stacks_2.npy', test_stacks_2)
    np.save('./data/' + args.data_name + '/test_stacks_3.npy', test_stacks_3)
    np.save('./data/' + args.data_name + '/test_gts.npy', test_gts)
    
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
    j = 0
    for i in gts_class:
        stack_index_i = stack_index[gts == i]
        gts_index_i = np.ones(n_shot, dtype=int)*j
        gts_index_i = gts_index_i[:, np.newaxis]    # 增加维度
        
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
    return epoch_stacks_1, epoch_stacks_2, epoch_stacks_3, epoch_gts