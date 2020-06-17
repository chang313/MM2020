# Made by Hyun Ryu, 2020/06/01
# CNN on guitar chord classification

import torch
import torch.nn as nn
import torch.nn.functional as F

# input: 3*1280*720 -> RGB channel
# output: 8 -> 8 classes

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool6 = nn.MaxPool2d(2, 2)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool7 = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(64*10*5, 320)
        self.linear2 = nn.Linear(320, 64)
        self.linear3 = nn.Linear(64, 14)
        
    
    def forward(self, x):
        out = self.maxpool1(F.relu(self.conv1(x)))
        out = self.maxpool2(F.relu(self.conv2(out)))
        out = self.maxpool3(F.relu(self.conv3(out)))
        out = self.maxpool4(F.relu(self.conv4(out)))
        out = self.maxpool5(F.relu(self.conv5(out)))
        out = self.maxpool6(F.relu(self.conv6(out)))
        out = self.maxpool7(F.relu(self.conv7(out)))
        
        out = out.view(-1, 64*10*5)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        
        return out



