import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import random_split, TensorDataset, DataLoader

'''
# data                |    keys    |        size       |      feature      |    class    |
# esardata.mat        |    3+1+1   |    (1300, 1200)   |        22         |     3+1     |
# esar_data.mat       |    3+5+1   |    (1300, 1200)   |   6,3,3,11,4(27)  |     3+1     |
# flevoland_data.mat  |    3+5+1   |     (750,1024)    |   6,3,3,11,4(27)  |     15+1    |
# san_data.mat        |    3+5+1   |     (900,1024)    |   6,3,3,11,4(27)  |     4+1     |

# esardata.mat : 'ESARfeat', 'ESARgrth'
# esar_data.mat : 'esar_t', 'esar_pau', 'esar_kro', 'esar_haa', 'esar_yam', 'esargrth'
# flevoland_data.mat : 'fle_t'6, 'fle_pau'3, 'fle_kro'3, 'fle_haa'11, 'fle_yam'4, 'flegrth'
# san_data.mat : 'san_t', 'san_pau', 'san_kro', 'san_haa', 'san_yam', 'sargrth'
background	255 255 255
labels 1-15
[252, 38, 38], [245, 130, 54], [253, 205, 11], [211, 254, 35], [109, 245, 19], 
[21, 244, 21], [40, 247, 123], [8, 253, 204], [56, 207, 245], [13, 108, 251], 
[33, 33, 245], [117, 30, 249], [206, 35, 249], [253, 50, 213], [248, 49, 129]
'''
def normalize(data):
    maxValue = torch.max(torch.max(data))
    minValue = torch.min(torch.min(data))
    data = (data - minValue) / (maxValue - minValue)
    return data

def mat_data(args):
    data_load = scipy.io.loadmat(args.data_folder + str(args.data_name) + '.mat')
    data_values = list(data_load.values())
    # key = list(data_load.keys())
    # 特征导入
    stack_t = torch.Tensor(data_values[3].astype(np.float32))
    stack_pau = torch.Tensor(data_values[4].astype(np.float32))
    stack_kro = torch.Tensor(data_values[5].astype(np.float32))
    stack_haa = torch.Tensor(data_values[6].astype(np.float32))
    stack_yam = torch.Tensor(data_values[7].astype(np.float32))
    stack = torch.cat([stack_t, stack_yam], 2)   
    # 根据需要的特征进行整合
    # stack = normalize(stack)
    # print('stack1: ', stack.shape)  # (750, 1024, 27)
    gt = torch.from_numpy(data_values[8])
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

    stacks_1 = torch.zeros(h_size * v_size, stack.shape[2], size_1 , size_1)
    stacks_2 = torch.zeros(h_size * v_size, stack.shape[2], size_2 , size_2)
    stacks_3 = torch.zeros(h_size * v_size, stack.shape[2], size_3 , size_3)
    gts = torch.zeros(h_size * v_size).int()

    i = 0
    for h in h_patches:
        for v in v_patches:
            for layer in range (stack.shape[2]):
                stacks_1[i][layer] = stack[(h-size_1//2):(h+size_1//2+1), (v-size_1//2):(v+size_1//2+1), layer]
                stacks_2[i][layer] = stack[(h-size_2//2):(h+size_2//2+1), (v-size_2//2):(v+size_2//2+1), layer]
                stacks_3[i][layer] = stack[(h-size_3//2):(h+size_3//2+1), (v-size_3//2):(v+size_3//2+1), layer]
            gts[i] = gt[h, v]
            i += 1
    print(f"stacks_1: {stacks_1.shape}")    # (739872,27,5,5)
    print(f"stacks_2: {stacks_2.shape}")    # (739872,27,11,11)
    print(f"stacks_3: {stacks_3.shape}")    # (739872,27,17,17)
    print(f"gts: {gts.shape}")      # (750360)
    return stacks_1, stacks_2, stacks_3, gts

def sar_datesets(args):
    stack, gt = mat_data(args)
    stack = normalize(stack)  # 特征归一化
    stacks_1, stacks_2, stacks_3, gts = reshape_rectangle_data(stack,gt,args.sar_size1,args.sar_size2,args.sar_size3)
    '''
    # 去掉背景
    index = torch.arange(0, gts.size(0))
    index_0 = index[gts != 0]
    stacks_1 = torch.index_select(stacks_1, 0, index_0) # (184156,27,5,5)
    stacks_2 = torch.index_select(stacks_2, 0, index_0)
    stacks_3 = torch.index_select(stacks_3, 0, index_0)
    gts = torch.index_select(gts, 0, index_0)   # (184156)
    '''
    print(f"Resizing image of size {stack.shape} to image patches {stacks_1.shape}, and grund-truth of size {gt.shape} to ground-truth patches {gts.shape}")

    num_class = torch.max(gts).numpy().astype(int)
    print("num_class: ", num_class)
    stacks_1 = stacks_1.numpy()
    stacks_2 = stacks_2.numpy()
    stacks_3 = stacks_3.numpy()
    gts = gts.numpy()

    
    np.save('./data/' + args.data_name + '/stacks_1.npy', stacks_1)
    np.save('./data/' + args.data_name + '/stacks_2.npy', stacks_2)
    np.save('./data/' + args.data_name + '/stacks_3.npy', stacks_3)
    np.save('./data/' + args.data_name + '/gts.npy', gts)
    
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
