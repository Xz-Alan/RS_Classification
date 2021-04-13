import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sar_data as sd
import os
import math
import time
import argparse
import scipy as sp
import scipy.stats
import scipy.io
from network import CNNEncoder, RelationNetwork
from sklearn import metrics
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
parser = argparse.ArgumentParser(description="remote sensing classification")
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--train_n_way", type=int, default=7)
parser.add_argument("--train_n_shot", type=int, default=5)
parser.add_argument("--train_n_query", type=int, default=15)
parser.add_argument("--test_n_way", type=int, default=7)
parser.add_argument("--test_n_shot", type=int, default=5)
parser.add_argument("--test_n_query", type=int, default=1)
parser.add_argument("--test_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--data_folder", type=str, default='./data/')
parser.add_argument("--data_name", type=str, default='sar_train_pool')
parser.add_argument("--sar_size1", type=int, default=5, help="flip the picture to 5x5 size")
parser.add_argument("--sar_size2", type=int, default=11, help="flip the picture to 11x11 size")
parser.add_argument("--sar_size3", type=int, default=17, help="flip the picture to 13x13 size")
parser.add_argument("--trainset_ratio", type=float, default=0.7)
parser.add_argument("--out_dim", type=int, default=32, help="cnn_net_out_dim")
parser.add_argument("--hidden_size", type=int, default=10, help="relation_net_hidden_size")
parser.add_argument("--loss_model", type=int, default=3, help="0: ce_loss;1: mse_loss;2: focal_loss;3: MSE_IIRL_loss")
parser.add_argument("--test_num", type=int, default=0)
parser.add_argument("--save_epoch", type=int, default=5000)
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
    encoded_indicies = torch.zeros(args.train_n_way*args.train_n_query, args.train_n_way).cuda()
    index = indices.long().view(-1,1)
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies



def main():
    print("loading %s_dataset", args.data_name)
    if os.path.exists('./data/' + args.data_name + '/train_stacks_1.npy') == False:
        print("making dataset")
        os.makedirs(("./data/"+args.data_name+"/"), exist_ok= True)
        sd.sar_datesets(args)
        print("make successful")
    
    train_stacks_1 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_1.npy')).unsqueeze(1)
    train_stacks_2 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_2.npy')).unsqueeze(1)
    train_stacks_3 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_3.npy')).unsqueeze(1)
    train_gts = torch.Tensor(np.load('./data/' + args.data_name + '/train_gts.npy'))
    print("stack3:", train_stacks_1.dtype)

    
    test_stacks_1 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_1.npy')).unsqueeze(1)
    test_stacks_2 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_2.npy')).unsqueeze(1)
    test_stacks_3 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_3.npy')).unsqueeze(1)
    test_gts = torch.Tensor(np.load('./data/' + args.data_name + '/test_gts.npy'))
    print("test: ", test_stacks_1.shape)
    print("gts: ", test_gts.shape)
    print("%sset load successfully"%(args.data_name))

    print("init network")
    cnn_sup = CNNEncoder(train_stacks_1.size(1), args.out_dim)
    cnn_que = CNNEncoder(train_stacks_1.size(1), args.out_dim)
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
    cnn_sup_scheduler = StepLR(cnn_sup_optim, step_size=50000, gamma=0.5)

    cnn_que_optim = torch.optim.Adam(cnn_que.parameters(), lr=args.lr)
    cnn_que_scheduler = StepLR(cnn_que_optim, step_size=50000, gamma=0.5)

    relation_net_optim = torch.optim.Adam(relation_net.parameters(), lr=args.lr)
    relation_net_scheduler = StepLR(relation_net_optim, step_size=50000, gamma=0.5)

    # prepare
    os.makedirs('./log', exist_ok=True)
    log = open("./log/%s_%d_loss_%d_shot_%d_log.txt"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num), 'w')
    log.write("{}\n".format(args))
    log.flush()
    cnn_sup_folder = "./model/" + str(args.data_name) + "/cnn_sup/"
    cnn_que_folder = "./model/" + str(args.data_name) + "/cnn_que/"
    relation_net_folder = "./model/" + str(args.data_name) + "/relation_net/"
    os.makedirs(cnn_sup_folder, exist_ok=True)
    os.makedirs(cnn_que_folder, exist_ok=True)
    os.makedirs(relation_net_folder, exist_ok=True)
    os.makedirs('./DataSave/', exist_ok=True)

    # checkpoint
    if os.path.exists(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_sup.load_state_dict(torch.load(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_sup successfully")
    if os.path.exists(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_que.load_state_dict(torch.load(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_que successfully")
    if os.path.exists(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)):
        relation_net.load_state_dict(torch.load(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load relation_net successfully")

    print("start training")
    num_iter = 5
    num_class = torch.max(train_gts).numpy().astype(int)
    print('The class numbers of the HSI data is:', num_class)
    KAPPA = []
    OA = []
    AA = []
    ELEMENT_ACC = np.zeros((num_iter, num_class))
    for index_iter in range(num_iter):
        pre_list = []
        labels_list = []
        for epoch in range(args.num_epoch):
            #--------------------------------------train--------------------------------------
            # train_dataloader
            gts_class = np.random.choice(np.arange(1,num_class+1), args.train_n_way, False)
            # nolabel_class = np.random.choice(np.arange(1,num_class+1), args.train_n_way, False)

            train_sup_stacks_1, train_sup_stacks_2, train_sup_stacks_3, train_sup_gts = sd.sar_dataloader(
                args, gts_class, train_gts, train_stacks_1, train_stacks_2, train_stacks_3, 
                split='train', form='support', shuffle=False)
            # (25,27,5,5/11,11/17,17)
            train_que_stacks_1, train_que_stacks_2, train_que_stacks_3, train_que_gts = sd.sar_dataloader(
                args, gts_class, train_gts, train_stacks_1, train_stacks_2, train_stacks_3, 
                split='train', form='query', shuffle=True)
            # (75,27,5,5/11,11/17,17)
            # nolabel_stacks, _ = sd.sar_dataloader(args, gts_class, test_gts, test_stacks, split='test', form='query', shuffle=True)

            train_sup_stacks_1 = train_sup_stacks_1.cuda()
            train_sup_stacks_2 = train_sup_stacks_2.cuda()
            train_sup_stacks_3 = train_sup_stacks_3.cuda()
            train_sup_gts = train_sup_gts.cuda()   # torch.arange(5).cuda()
            train_que_stacks_1 = train_que_stacks_1.cuda()
            train_que_stacks_2 = train_que_stacks_2.cuda()
            train_que_stacks_3 = train_que_stacks_3.cuda()
            train_que_gts = train_que_gts.cuda()
            # nolabel_stacks = nolabel_stacks.cuda()

            mult_sup_feature = cnn_sup(train_sup_stacks_1, train_sup_stacks_2, train_sup_stacks_3)  # tuple: (25,32,2,2/5,5/8,8)
            mult_que_feature = cnn_que(train_que_stacks_1, train_que_stacks_2, train_que_stacks_3)
            # nolabel_feature = cnn_sup(nolabel_stacks)   # (5,32,5,5)
            #-------------------------------------sup&que-------------------------------------
            mult_relation_pairs = []
            for i in range(3):
                # 支持集特征按类取平均
                sup_feature = mult_sup_feature[i]
                que_feature = mult_que_feature[i]
                sup_feature = sup_feature.view(args.train_n_way, args.train_n_shot, -1, sup_feature.shape[2], sup_feature.shape[3])   # (5,5,32,5,5)
                sup_feature = torch.mean(sup_feature,1).squeeze(1)     #(5,32,5,5)
                # print("sup_feature: ", sup_feature.shape)
                # relation拼接
                sup_feature_ext = sup_feature.unsqueeze(0).repeat(args.train_n_way*args.train_n_query, 1, 1, 1, 1)  # (75,5,32,5,5)
                # print("sup_feature_ext: ", sup_feature_ext.shape)
                que_feature_ext = torch.transpose(que_feature.unsqueeze(0).repeat(args.train_n_way,1,1,1,1),0,1)     # (5,75,32,5,5)-->(75,5,32,5,5)
                # print("que_feature_ext: ", que_feature_ext.shape)

                relation_pairs = torch.cat((sup_feature_ext, que_feature_ext), 2)
                relation_pairs = relation_pairs.view(-1, 2*args.out_dim, sup_feature.shape[2], sup_feature.shape[3])   # (75,5,64,5,5)-->(375,64,5,5)
                # print("relation_pairs: ", relation_pairs.shape)
                mult_relation_pairs.append(relation_pairs)
            # print("1: ", mult_relation_pairs[0].shape)
            # print("2: ", mult_relation_pairs[1].shape)
            # print("3: ", mult_relation_pairs[2].shape)
            
            relations = relation_net(mult_relation_pairs[0], mult_relation_pairs[1], mult_relation_pairs[2]).view(-1, args.train_n_way)     # (375,1)-->(75,5)
            _, predict_gts = torch.max(relations.data, 1)
            pre_list.extend(np.array(predict_gts.cpu().int()))
            labels_list.extend(np.array(train_que_gts.cpu().int()))
        
        confusion_matrix = metrics.confusion_matrix(pre_list, labels_list)
        overall_acc = metrics.accuracy_score(pre_list, labels_list)
        kappa = metrics.cohen_kappa_score(pre_list, labels_list)
        each_acc, average_acc = utils.aa_and_each_accuracy(confusion_matrix)

        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        ELEMENT_ACC[index_iter, :] = each_acc
    print("OA: ", OA, "\nAA: ", AA, "\nKappa: ", KAPPA)
    utils.record_output(OA, AA, KAPPA, ELEMENT_ACC, 
                        './records/' + args.data_name + '.txt')


if __name__ == "__main__":
    main()

'''
1 :  248612
2 :  93139
3 :  283190
4 :  38432
5 :  89642
6 :  54647
7 :  2338
'''