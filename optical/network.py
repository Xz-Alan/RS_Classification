import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, in_channel, out_channel):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
                        # nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,out_channel,kernel_size=3,padding=1),
                        nn.BatchNorm2d(out_channel, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x_1,x_2,x_3):
        out_1 = self.layer1(x_1)
        out_1 = self.layer2(out_1)
        out_1 = self.layer3(out_1)
        
        out_2 = self.layer1(x_2)    # (25,27,11,11)-->(25,64,5,5)
        out_2 = self.layer2(out_2)
        out_2 = self.layer3(out_2)  # (25,64,5,5)-->(25,32,5,5)

        out_3 = self.layer1(x_3)
        out_3 = self.layer2(out_3)
        out_3 = self.layer3(out_3)
        return out_1, out_2, out_3 

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,in_channel,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channel,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,32,kernel_size=3,padding=1),
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        )
        self.fc1 = nn.Sequential(nn.Linear(32,hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x_1,x_2,x_3):
        p_1 = self.layer1(x_3) / 2 + x_2 / 2    # (375,64,11,11)
        p_2 = self.layer2(p_1) / 2 + x_1 / 2    # (375,64,5,5)

        out = self.layer3(p_2)      # (375,64,2,2)
        out = self.layer4(out)      # (375,32,1,1)
        
        out = out.view(out.size(0),-1)  # (375,32)
        out = self.fc1(out)     # (375,10)
        out = self.fc2(out)     # (375,1)
        return out