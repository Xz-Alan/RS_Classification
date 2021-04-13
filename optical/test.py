import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sar_data as sd
import test_sar_data as tsd
import os
import math
import time
import argparse
import scipy as sp
import scipy.stats
import scipy.io
from PIL import Image
import random
from network import CNNEncoder, RelationNetwork
from sklearn.metrics import confusion_matrix
import rgb

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description="hsi few-shot classification")
parser.add_argument("--num_epoch", type=int, default=1)
parser.add_argument("--train_n_way", type=int, default=7)
parser.add_argument("--train_n_shot", type=int, default=5)
parser.add_argument("--train_n_query", type=int, default=15)
parser.add_argument("--test_n_way", type=int, default=7)
parser.add_argument("--test_n_shot", type=int, default=5)
parser.add_argument("--test_n_query", type=int, default=1)
parser.add_argument("--test_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--data_folder", type=str, default='./data/')
parser.add_argument("--data_name", type=str, default='rs_data')    # flevoland
parser.add_argument("--sar_size1", type=int, default=5, help="flip the picture to 5x5 size")
parser.add_argument("--sar_size2", type=int, default=11, help="flip the picture to 11x11 size")
parser.add_argument("--sar_size3", type=int, default=17, help="flip the picture to 13x13 size")
parser.add_argument("--trainset_ratio", type=float, default=0.7)
parser.add_argument("--out_dim", type=int, default=32, help="cnn_net_out_dim")
parser.add_argument("--hidden_size", type=int, default=10, help="relation_net_hidden_size")
parser.add_argument("--loss_model", type=int, default=3, help="0: ce_loss;1: mse_loss;2: focal_loss;3: MSE_IIRL_loss")
parser.add_argument("--test_num", type=int, default=0)
parser.add_argument("--test_switch",type=bool, default=False)
parser.add_argument("--paint_switch",type=bool,default=False)
args = parser.parse_args()

def weights_init(m):
    """
    initial model.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def one_hot(args, indices):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    """
    encoded_indicate = torch.zeros(args.train_n_way*args.train_n_query, args.train_n_way).cuda()
    index = indices.long().view(-1,1)
    encoded_indicate = encoded_indicate.scatter_(1,index,1)
    
    return encoded_indicate


def kappa(confusion_matrix):
    """kappa系数
    :param: confusion_matrix--混淆矩阵
    :return: Kappa系数
    """
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)


def main():
    
    rgb_colors = rgb.ncolors(args.train_n_way)
    print(rgb_colors)
    
    start_time = time.time()
    # rgb_colors = np.array([[248, 49, 49], [200, 248, 9], [42, 248, 124], [36, 123, 254], [204, 4, 254]])
    if args.paint_switch:
        print("painting img_gt")
        _, gts = sd.mat_data(args)
        wait
        gts -= 1
        img_h = gts.shape[0]-16
        img_v = gts.shape[1]-16
        img_gt = Image.new("RGB", (img_h, img_v), "white")
        for h in range(img_h):
            for v in range(img_v):
                for i in range(args.test_n_way):
                    if gts[h+8,v+8] == i:
                        img_gt.putpixel([h, v], (rgb_colors[i][0], rgb_colors[i][1], rgb_colors[i][2]))
                        break
        img_gt.save("./img_result/"+ str(args.data_name) + "_img_gt.jpg")
    

    if args.test_switch:
        # 184170 load
        que_labels = scipy.io.loadmat("./labels_save/que_%s_%d_loss_%d_shot_%d_img_out.mat"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num))['que_labels'].squeeze(0).astype(int)
        pre_labels = scipy.io.loadmat("./labels_save/pre_%s_%d_loss_%d_shot_%d_img_out.mat"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num))['pre_labels'].squeeze(0)
        # perpare
        class_correct = np.zeros(args.test_n_way).astype(int)
        class_num = np.zeros(args.test_n_way).astype(int)
        class_acc = np.zeros(args.test_n_way).astype(float)
        for i in range(len(que_labels)):
            if pre_labels[i]==que_labels[i]:
                class_correct[que_labels[i]] += 1
            class_num[que_labels[i]] += 1
        # kappa
        confusion_m = confusion_matrix(que_labels, pre_labels)
        kappa_score = kappa(confusion_m)
        print("Kappa: %.2f %%" %(kappa_score*100))
        # aa
        for i in range(args.test_n_way):
            class_acc[i] = class_correct[i] / class_num[i]
            print("class_%d_acc: %.2f %%" %(i, class_acc[i]*100))
        aa = np.mean(class_acc)
        print("AA: %.2f %%" %(aa*100))
        # oa
        total_labels = np.sum(class_num)
        total_correct = np.sum(class_correct)
        oa = total_correct/1.0 / total_labels/1.0
        print("OA: %.2f %%" %(oa*100))
        return print("test finished!")
    
    print("loading sar_dataset")
    if os.path.exists('./data/' + args.data_name + '/stacks_1.npy') == False:
        print("making dataset")
        os.makedirs(("./data/"+args.data_name+"/"), exist_ok= True)
        tsd.sar_datesets(args)
    
    test_stacks_1 = torch.Tensor(np.load('./data/' + args.data_name + '/stacks_1.npy'))    # (182656,27,5,5)
    test_stacks_2 = torch.Tensor(np.load('./data/' + args.data_name + '/stacks_2.npy'))
    test_stacks_3 = torch.Tensor(np.load('./data/' + args.data_name + '/stacks_3.npy'))
    test_gts = torch.Tensor(np.load('./data/' + args.data_name + '/gts.npy'))
    test_gts -= 1
    load_time = time.time()
    print("%sset load successfully, and spend time: %.2f"%(args.data_name, load_time-start_time))
    
    print("init network")
    cnn_sup = CNNEncoder(test_stacks_1.size(1), args.out_dim)
    cnn_que = CNNEncoder(test_stacks_1.size(1), args.out_dim)
    relation_net = RelationNetwork(2*args.out_dim, args.hidden_size)
    # 初始化模型
    cnn_sup.apply(weights_init)
    cnn_que.apply(weights_init)
    relation_net.apply(weights_init)
    
    cnn_sup.cuda()
    cnn_que.cuda()
    relation_net.cuda()

    # scheduler
    # Adam 对网络参数进行优化，学习率10000次循环后降为原来的0.5倍
    cnn_sup_optim = torch.optim.Adam(cnn_sup.parameters(), lr=args.lr)
    cnn_sup_scheduler = StepLR(cnn_sup_optim, step_size=20000, gamma=0.5)

    cnn_que_optim = torch.optim.Adam(cnn_que.parameters(), lr=args.lr)
    cnn_que_scheduler = StepLR(cnn_que_optim, step_size=20000, gamma=0.5)

    relation_net_optim = torch.optim.Adam(relation_net.parameters(), lr=args.lr)
    relation_net_scheduler = StepLR(relation_net_optim, step_size=20000, gamma=0.1)

    test_result = open("./test_result/%s_%d_loss_%d_shot_%d_log.txt"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num), 'w')
    cnn_sup_folder = "./model/" + str(args.data_name) + "/cnn_sup/"
    cnn_que_folder = "./model/" + str(args.data_name) + "/cnn_que/"
    relation_net_folder = "./model/" + str(args.data_name) + "/relation_net/"
    os.makedirs(cnn_sup_folder, exist_ok=True)
    os.makedirs(cnn_que_folder, exist_ok=True)
    os.makedirs(relation_net_folder, exist_ok=True)

    if os.path.exists(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_sup.load_state_dict(torch.load(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_sup successfully")
    if os.path.exists(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_que.load_state_dict(torch.load(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_que successfully")
    if os.path.exists(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        relation_net.load_state_dict(torch.load(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load relation_net successfully")
    '''
    cnn_sup.eval()
    cnn_que.eval()
    relation_net.eval()
    '''

    for epoch in range(args.num_epoch):
        print("start testing")
        #------------------------------prepare------------------------------
        test_time = time.time()
        total_correct = 0
        class_correct = np.zeros(args.test_n_way).astype(int)
        class_acc = np.zeros(args.test_n_way).astype(float)
        pre_labels = []
        que_labels = []
        gts_class = np.arange(args.test_n_way)
        h_img = 750 -16
        v_img = 1024 -16
        img_out = Image.new("RGB", (h_img, v_img), "white")
        #------------------------------test------------------------------
        test_sup_stacks_1, test_sup_stacks_2, test_sup_stacks_3, test_sup_gts, class_num = tsd.sar_dataloader(args, gts_class, test_gts, test_stacks_1, test_stacks_2, test_stacks_3, split='test',form='support', shuffle=False)

        class_num_max = np.max(class_num)
        print("class_num_max: ", class_num_max)
        index_i = np.zeros(args.test_n_way).astype(int)
        index_j = np.zeros(args.test_n_way).astype(int)
        for i in range(class_num_max):
            #-------------------------------------------------------------------------
            stack_index = np.arange(0, test_gts.size(0))   # 生成stack的索引
            # print("stack_index: ", len(stack_index))
            index = np.zeros(1, dtype=int)    # 生成一个零数组，方便for循环
            for i in gts_class:
                stack_index_i = stack_index[test_gts == i]
                if index_j[i] >= len(stack_index_i):
                    index_j[i] = 0
                # print(i, ":", len(stack_index_i))
                stack_index_i = [stack_index_i[index_j[i]]]
                index = np.concatenate((index, stack_index_i), axis=0)
                index_j[i] += 1
            index = np.delete(index, 0 , 0)     # 不打乱顺序
            test_que_stacks_1 = []
            test_que_stacks_2 = []
            test_que_stacks_3 = []
            test_que_gts = []
            for item in list(index):
                # 每一行需要增加一维，拼接时保证维度正确
                test_que_stacks_1.append(test_stacks_1[item].unsqueeze(0))
                test_que_stacks_2.append(test_stacks_2[item].unsqueeze(0))
                test_que_stacks_3.append(test_stacks_3[item].unsqueeze(0))
                test_que_gts.append(test_gts[item].unsqueeze(0))
            test_que_stacks_1 = torch.cat(test_que_stacks_1, dim=0)   # (25,27,5,5)
            test_que_stacks_2 = torch.cat(test_que_stacks_2, dim=0)   # (25,27,11,11)
            test_que_stacks_3 = torch.cat(test_que_stacks_3, dim=0)   # (25,27,17,17)
            test_que_gts = torch.cat(test_que_gts, dim=0)
            #-------------------------------------------------------------------------
            test_sup_stacks_1 = test_sup_stacks_1.cuda()
            test_sup_stacks_2 = test_sup_stacks_2.cuda()
            test_sup_stacks_3 = test_sup_stacks_3.cuda()
            test_sup_gts = test_sup_gts.cuda()
            test_que_stacks_1 =  test_que_stacks_1.cuda()
            test_que_stacks_2 =  test_que_stacks_2.cuda()
            test_que_stacks_3 =  test_que_stacks_3.cuda()
            test_que_gts = test_que_gts.cuda()

            mult_sup_feature = cnn_sup(test_sup_stacks_1, test_sup_stacks_2, test_sup_stacks_3)
            mult_que_feature = cnn_que(test_que_stacks_1, test_que_stacks_2, test_que_stacks_3)

            mult_relation_pairs = []
            for i in range(3):
                # 支持集按类取平均
                sup_feature = mult_sup_feature[i]
                que_feature = mult_que_feature[i]
                sup_feature = sup_feature.view(args.test_n_way, args.test_n_shot, -1, sup_feature.shape[2], sup_feature.shape[3])
                sup_feature = torch.mean(sup_feature,1).squeeze(1)

                # relations
                sup_feature_ext = sup_feature.unsqueeze(0).repeat(args.test_n_way*args.test_n_query, 1, 1, 1, 1)
                que_feature_ext = torch.transpose(que_feature.unsqueeze(0).repeat(args.test_n_way,1,1, 1, 1),0,1)

                relation_pairs = torch.cat((sup_feature_ext, que_feature_ext), 2).view(-1, 2*args.out_dim, sup_feature.shape[2], sup_feature.shape[3])
                mult_relation_pairs.append(relation_pairs)
                
            relations = relation_net(mult_relation_pairs[0], mult_relation_pairs[1], mult_relation_pairs[2]).view(-1, args.test_n_way)
            # calculate relations
            _, predict_gts = torch.max(relations.data, 1)
            for j in range(args.test_n_way):
                h_j = index[j] // v_img
                v_j = index[j] % v_img
                img_out.putpixel([h_j, v_j], (rgb_colors[predict_gts[j]][0], rgb_colors[predict_gts[j]][1], rgb_colors[predict_gts[j]][2]))
                if index_i[j] > class_num[j]:
                    continue
                if predict_gts[j]== test_que_gts[j]:
                    class_correct[j] += 1
                pre_labels.append(predict_gts[j].item())
                que_labels.append(test_que_gts[j].item())
                index_i[j] +=1
        # painting
        img_out.save("./img_result/" + "%s_%d_loss_%d_shot_%d_img_out.jpg"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num))
        # labels save
        que_save = "./labels_save/que_%s_%d_loss_%d_shot_%d_img_out.mat"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)
        pre_save = "./labels_save/pre_%s_%d_loss_%d_shot_%d_img_out.mat"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)
        scipy.io.savemat(que_save, mdict={"que_labels": que_labels})
        scipy.io.savemat(pre_save, mdict={"pre_labels": pre_labels})
        # kappa
        confusion_m = confusion_matrix(que_labels, pre_labels)
        kappa_score = kappa(confusion_m)
        print("Kappa: %.2f %%" %(kappa_score*100))
        test_result.write("Kappa: %.2f %%\n" %(kappa_score*100))
        test_result.flush()
        # aa
        for i in range(args.test_n_way):
            class_acc[i] = class_correct[i] / class_num[i]
            # print(i, "_class_correct: ", class_correct[i])
            # print(i, "_class_num: ", class_num[i])
            print("class_%d_acc: %.2f %%" %(i, class_acc[i]*100))
            test_result.write("class_%d_acc: %.2f %%\n" %(i, class_acc[i]*100))
            test_result.flush()
        aa = np.mean(class_acc)
        print("AA: %.2f %%" %(aa*100))
        test_result.write("AA: %.2f %%\n" %(aa*100))
        test_result.flush()
        # oa
        total_labels = np.sum(class_num)
        total_correct = np.sum(class_correct)
        # print("total_labels: ", total_labels)
        # print("total_correct: ", total_correct)
        oa = total_correct / total_labels
        print("OA: %.2f %%" %(oa*100))
        test_result.write("OA: %.2f %%\n" %(oa*100))
        test_result.flush()
        end_time = time.time()
        print("test finished, and spend time: ", end_time - test_time)
        
if __name__ == "__main__":
    main()
