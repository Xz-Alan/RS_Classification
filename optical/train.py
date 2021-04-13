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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description="remote sensing classification")
parser.add_argument("--num_epoch", type=int, default=100001)
parser.add_argument("--train_n_way", type=int, default=7)
parser.add_argument("--train_n_shot", type=int, default=5)
parser.add_argument("--train_n_query", type=int, default=15)
parser.add_argument("--test_n_way", type=int, default=7)
parser.add_argument("--test_n_shot", type=int, default=5)
parser.add_argument("--test_n_query", type=int, default=1)
parser.add_argument("--test_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--data_folder", type=str, default='./data/')
parser.add_argument("--data_name", type=str, default='optical_train_pool')
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
    stack = sd.mat_data(args)
    print("loading sar_dataset")
    if os.path.exists('./data/' + args.data_name + '/train_stacks_1.npy') == False:
        print("making dataset")
        os.makedirs(("./data/"+args.data_name+"/"), exist_ok= True)
        sd.sar_datesets(args)
        print("make successful")
    
    train_stacks_1 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_1.npy'))  # (1500,27,5,5)
    train_stacks_2 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_2.npy'))  
    train_stacks_3 = torch.Tensor(np.load('./data/' + args.data_name + '/train_stacks_3.npy'))
    print("stack3:", train_stacks_1.dtype)

    train_gts = torch.Tensor(np.load('./data/' + args.data_name + '/train_gts.npy'))
    test_stacks_1 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_1.npy'))    # (182656,27,5,5)
    test_stacks_2 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_2.npy'))
    test_stacks_3 = torch.Tensor(np.load('./data/' + args.data_name + '/test_stacks_3.npy'))
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
    log = open("./log/%s_%d_loss_%d_shot_%d_log.txt"%(args.data_name, args.loss_model, args.train_n_shot, args.test_num), 'a')  # 追加不覆盖
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
    last_acc = 0.0
    loss_list = []
    acc_list = []
    prob = []
    for epoch in range(args.num_epoch):
        
        #--------------------------------------train--------------------------------------
        # train_dataloader
        num_class = torch.max(train_gts).numpy().astype(int)
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
        
        # print("relations: ", relations.shape)
        # _, predict_gts = torch.max(relations.data, 1)

        #-------------------------------------CE_loss-------------------------------------
        if args.loss_model == 0:
            ce_loss = nn.CrossEntropyLoss().cuda()
            loss = ce_loss(relations, train_que_gts.long()) # + loss_nolabel
            # print("loss1: ", loss)
        
        #-------------------------------------MSE_loss-------------------------------------
        elif args.loss_model == 1:
            mse = nn.MSELoss().cuda()
            softmax = nn.Softmax(dim=1).cuda()
            relations_softmax = softmax(relations)
            train_que_gts_one_hot = one_hot(args, train_que_gts)
            # print("train_que_gts_one_hot: ", train_que_gts_one_hot.shape)
            loss = mse(relations_softmax, train_que_gts_one_hot)    # + loss_nolabel
            # print("loss: ", loss)
        #-------------------------------------Focal_loss-------------------------------------
        elif args.loss_model == 2:
            softmax = nn.Softmax(dim=1).cuda()
            relations_softmax = softmax(relations)
            train_que_gts_one_hot = one_hot(args, train_que_gts)
            p_t = train_que_gts_one_hot.mul(relations_softmax) + (1 - train_que_gts_one_hot).mul(1 - relations_softmax)
            focal_matrix = -((1 - p_t).mul(1 - p_t)).mul(torch.log(p_t))
            loss = torch.sum(torch.sum(focal_matrix)) / relations.shape[1]  # + loss_nolabel
        #-------------------------------------MSE_IIRL_loss-------------------------------------
        elif args.loss_model == 3:
            margin_thr = 1.5
            # margin_thr2 = 0.5
            softmax = nn.Softmax(dim=1).cuda()
            mse = nn.MSELoss().cuda()
            ce_loss = nn.CrossEntropyLoss().cuda()
            relations_softmax = softmax(relations)
            # print("relations_softmax: ", relations_softmax.shape)
            train_que_gts_one_hot = one_hot(args, train_que_gts)

            # diff_que = torch.sum(train_que_gts_one_hot.mul(relations_softmax),1,True).repeat(1,5) - relations_softmax - margin_thr2 * (1-train_que_gts_one_hot)  # (75,5)
            pos_que = torch.sum(train_que_gts_one_hot.mul(relations_softmax),1,True)    # (75,1)
            neg_que = torch.sum((relations_softmax - train_que_gts_one_hot.mul(relations_softmax)),1,True) / (args.train_n_way - 1) # (75,1)
            diff_que = (pos_que / neg_que) - margin_thr * torch.sum(train_que_gts_one_hot,1,True)

            loss_diff_que = torch.sum(torch.log(1-torch.where(diff_que >=0, torch.full_like(diff_que, 0), diff_que)), 1, True) / relations_softmax.shape[0]
            loss_label = ce_loss(relations, train_que_gts.long())
            # loss_label = mse(relations_softmax, train_que_gts_one_hot)
            
            loss = torch.mean(loss_diff_que + loss_label)   # + 1.0 / 4 * loss_nolabel
            # print("loss1: ", loss)

        #-------------------------------------loss_backward-------------------------------------
        cnn_sup.zero_grad()
        cnn_que.zero_grad()
        relation_net.zero_grad()

        loss.backward()
        # print("loss2: ", loss)
        cnn_sup_optim.step()
        cnn_que_optim.step()
        relation_net_optim.step()

        cnn_sup_scheduler.step()
        cnn_que_scheduler.step()
        relation_net_scheduler.step()
        #--------------------------------------test--------------------------------------
        if (epoch+1) % 500 == 0:
            print("start testing")
            total_correct = 0
            for epoch_test in range(args.test_epoch):
                gts_class = np.random.choice(np.arange(1,num_class+1), args.test_n_way, False)

                test_sup_stacks_1, test_sup_stacks_2, test_sup_stacks_3, test_sup_gts = sd.sar_dataloader(
                    args, gts_class, test_gts, test_stacks_1, test_stacks_2, test_stacks_3,
                    split='test',form='support', shuffle=False)
                for e in range(5):
                    test_que_stacks_1, test_que_stacks_2, test_que_stacks_3, test_que_gts = sd.sar_dataloader(
                        args, gts_class, test_gts, test_stacks_1, test_stacks_2, test_stacks_3, split='test',form='query', shuffle=True)
                    
                    batch = test_que_gts.shape[0]

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
                    prob.append(relations.data)
                    # calculate relations
                    _, predict_gts = torch.max(relations.data, 1)
                    correct = [1 if predict_gts[j]== test_que_gts[j] else 0 for j in range(batch)]
                    total_correct += np.sum(correct)
                # print("total_correct: ", total_correct)
            
            test_acc = total_correct/1.0/(args.test_n_way*args.test_epoch*5/1.0)
            print("epoch: %d, test_acc: %.2f %%, loss: %.4f" %(epoch, test_acc*100, loss.item()))
            epoch_log = "epoch: %d, test_acc: %.2f %%, loss: %.4f\n" %(epoch, test_acc*100, loss.item())
            log.write(epoch_log)
            log.flush()
            acc_list.append(test_acc)
            loss_list.append(loss.item())

            if test_acc > last_acc:
                torch.save(cnn_sup.state_dict(), cnn_sup_folder + "%s_%d_loss_%d_shot_%d.pth" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num))
                torch.save(cnn_que.state_dict(), cnn_que_folder + "%s_%d_loss_%d_shot_%d.pth" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num))
                torch.save(relation_net.state_dict(), relation_net_folder + "%s_%d_loss_%d_shot_%d.pth" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num))
                print("save networks for epoch: ", epoch)
                last_acc = test_acc

        if epoch == args.num_epoch - 1:
            print("train finished, and best acc is %.2f %%" %(last_acc*100))
            epoch_log = "train finished, and best acc is %.2f %%\n" %(last_acc*100)
            log.write(epoch_log)
            log.flush()

        if epoch % args.save_epoch == 0 :
            loss_save_name = "DataSave/loss_%s_%d_loss_%d_shot_%d_%d_epoch.mat" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num, epoch)
            acc_save_name = "DataSave/acc_%s_%d_loss_%d_shot_%d_%d_epoch.mat" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num, epoch)
            scipy.io.savemat(loss_save_name, mdict={"loss_list": loss_list})
            scipy.io.savemat(acc_save_name, mdict={"acc_list": acc_list})
            print("save loss and acc successfully")
            loss_save_name_before = "DataSave/loss_%s_%d_loss_%d_shot_%d_%d_epoch.mat" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num, epoch - args.save_epoch)
            acc_save_name_before = "DataSave/acc_%s_%d_loss_%d_shot_%d_%d_epoch.mat" %(args.data_name, args.loss_model, args.train_n_shot, args.test_num, epoch - args.save_epoch)
            if os.path.exists(loss_save_name_before):
                os.remove(loss_save_name_before)
            if os.path.exists(acc_save_name_before):
                os.remove(acc_save_name_before)

if __name__ == "__main__":
    main()
