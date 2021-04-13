import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sar_data as sd
import test_sar_data as tsd
import all_sar_data as asd
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
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description="remote sensing classification")
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
parser.add_argument("--sup_data_name", type=str, default='sar_train_pool')
parser.add_argument("--que_data_name", type=str, default='sar_test_pool')
parser.add_argument("--sar_size1", type=int, default=5, help="flip the picture to 5x5 size")
parser.add_argument("--sar_size2", type=int, default=11, help="flip the picture to 11x11 size")
parser.add_argument("--sar_size3", type=int, default=17, help="flip the picture to 13x13 size")
parser.add_argument("--trainset_ratio", type=float, default=0.7)
parser.add_argument("--out_dim", type=int, default=32, help="cnn_net_out_dim")
parser.add_argument("--hidden_size", type=int, default=10, help="relation_net_hidden_size")
parser.add_argument("--loss_model", type=int, default=3, help="0: ce_loss;1: mse_loss;2: focal_loss;3: MSE_IIRL_loss")
parser.add_argument("--test_num", type=int, default=0)
parser.add_argument("--epoch_sup", type=int, default=17)
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
    print(args.que_data_name)
    input()
    start_time = time.time()
    stack = asd.mat_data(args)
    h_img = stack.shape[0] -16
    v_img = stack.shape[1] -16
    # rgb_colors = rgb.ncolors(args.train_n_way)
    # print(rgb_colors)
    rgb_colors = np.array([[255, 0, 0], [255, 255, 0], [64, 0, 0], [138,43,226], [0, 255, 0], [0, 64, 0], [0, 128, 255]])
    
    print("loading sar_dataset")
    if os.path.exists('./data/' + args.que_data_name + '/stacks_1.npy') == False:
        print("making dataset")
        os.makedirs(("./data/"+args.que_data_name+"/"), exist_ok= True)
        asd.sar_datesets(args)
    
    sup_stacks_1 = torch.Tensor(np.load('./data/' + args.sup_data_name + '/stacks_1.npy')).unsqueeze(1)
    sup_stacks_2 = torch.Tensor(np.load('./data/' + args.sup_data_name + '/stacks_2.npy')).unsqueeze(1)
    sup_stacks_3 = torch.Tensor(np.load('./data/' + args.sup_data_name + '/stacks_3.npy')).unsqueeze(1)
    sup_gts = torch.Tensor(np.load('./data/' + args.sup_data_name + '/gts.npy'))
    sup_gts -= 1
    print(sup_stacks_1.shape)
    # print(torch.min(sup_gts))
    # print(torch.max(sup_gts))
    que_stacks_1 = torch.Tensor(np.load('./data/' + args.que_data_name + '/stacks_1.npy')).unsqueeze(1)
    que_stacks_2 = torch.Tensor(np.load('./data/' + args.que_data_name + '/stacks_2.npy')).unsqueeze(1)
    que_stacks_3 = torch.Tensor(np.load('./data/' + args.que_data_name + '/stacks_3.npy')).unsqueeze(1)
    # que_gts = torch.Tensor(np.load('./data/' + args.que_data_name + '/gts.npy'))
    print(que_stacks_1.shape)

    load_time = time.time()
    print("%sset and %sset load successfully, and spend time: %.2f"%(args.sup_data_name, args.que_data_name, load_time-start_time))
    
    print("init network")
    cnn_sup = CNNEncoder(que_stacks_1.size(1), args.out_dim)
    cnn_que = CNNEncoder(que_stacks_1.size(1), args.out_dim)
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

    cnn_sup_folder = "./model/" + str(args.sup_data_name) + "/cnn_sup/"
    cnn_que_folder = "./model/" + str(args.sup_data_name) + "/cnn_que/"
    relation_net_folder = "./model/" + str(args.sup_data_name) + "/relation_net/"
    os.makedirs(cnn_sup_folder, exist_ok=True)
    os.makedirs(cnn_que_folder, exist_ok=True)
    os.makedirs(relation_net_folder, exist_ok=True)
    os.makedirs('./labels_save/', exist_ok=True)
    os.makedirs('./img_result/', exist_ok=True)

    if os.path.exists(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_sup.load_state_dict(torch.load(cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_sup successfully")
    if os.path.exists(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)):
        cnn_que.load_state_dict(torch.load(cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load cnn_que successfully")
    if os.path.exists(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)):
        relation_net.load_state_dict(torch.load(relation_net_folder + "%s_%d_loss_%d_shot_%d.pth"%(args.sup_data_name, args.loss_model, args.train_n_shot, args.test_num)))
        print("load relation_net successfully")
    '''
    cnn_sup.eval()
    cnn_que.eval()
    relation_net.eval()
    '''
    # input("开始训练：")
    for epoch in range(args.num_epoch):
        print("start testing")
        #------------------------------prepare------------------------------
        test_time = time.time()
        pre_labels = []
        gts_class = np.arange(args.test_n_way)
        
        epoch_que = h_img * v_img // args.train_n_way
        # print(epoch_que)
        epoch_que /= args.epoch_sup
        # print(epoch_que)
        index = np.arange(que_stacks_1.shape[0])
        np.random.seed(1012)
        np.random.shuffle(index)
        index_epoch = 0

        img_out = Image.new("RGB", (h_img, v_img), "white")
        #------------------------------test------------------------------
        for sup_epoch in range(args.epoch_sup):

            test_sup_stacks_1, test_sup_stacks_2, test_sup_stacks_3, test_sup_gts, class_num = asd.sar_dataloader(args, gts_class, sup_gts, sup_stacks_1, sup_stacks_2, sup_stacks_3, split='test',form='support', shuffle=False)
            # print("sup: ", test_sup_stacks_1.shape)
            
            for que_epoch in tqdm(range(int(epoch_que))):
                #-------------------------------------------------------------------------
                index_select = np.arange(index_epoch, index_epoch + args.train_n_way)
                # print(index_select)
                test_que_stacks_1 = que_stacks_1[index[index_select]]
                test_que_stacks_2 = que_stacks_2[index[index_select]]
                test_que_stacks_3 = que_stacks_3[index[index_select]]
                # print("que: ", test_que_stacks_1.shape)
                # print("index: ", index[index_select])
                #-------------------------------------------------------------------------
                test_sup_stacks_1 = test_sup_stacks_1.cuda()
                test_sup_stacks_2 = test_sup_stacks_2.cuda()
                test_sup_stacks_3 = test_sup_stacks_3.cuda()
                test_sup_gts = test_sup_gts.cuda()
                test_que_stacks_1 =  test_que_stacks_1.cuda()
                test_que_stacks_2 =  test_que_stacks_2.cuda()
                test_que_stacks_3 =  test_que_stacks_3.cuda()

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
                # print("pre: ", predict_gts)
                #input()
                for j in range(args.train_n_way):
                    h_j = index[index_select][j] // v_img
                    v_j = index[index_select][j] % v_img
                    img_out.putpixel([h_j, v_j], (rgb_colors[predict_gts[j]][0], rgb_colors[predict_gts[j]][1], rgb_colors[predict_gts[j]][2]))
                    pre_labels.append(predict_gts[j].item())
                index_epoch += args.train_n_way
            # painting
            img_out.save("./img_result/" + "%s_%d_loss_%d_shot_%d_img_out.jpg"%(args.que_data_name, args.loss_model, args.train_n_shot, args.test_num))
            # input()

        # labels save
        pre_save = "./labels_save/pre_%s_%d_loss_%d_shot_%d_img_out.mat"%(args.que_data_name, args.loss_model, args.train_n_shot, args.test_num)
        scipy.io.savemat(pre_save, mdict={"pre_labels": pre_labels})
        
        end_time = time.time()
        print("test finished, and spend time: ", end_time - test_time)
        
if __name__ == "__main__":
    main()
