# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:38:51 2020

@author: zhang_2020
"""


#加载mnist数据》pickle
from pathlib import Path
import pickle
import gzip

PATH = Path(r"D:\ZHANG2020\PytorchX\data\mnist")
FILENAME = "mnist.pkl.gz"

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


#默认为numpy格式》转换为Tensor》转换训练集》查看单个样本
import torch 
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

BATCH_SIZE = 10

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

data = iter(train_dl)
xb, yb = data.next()
print("xb = ",xb)
print("yb = ",yb)
    
# 定于Model》nn.Sequential()》Custom Layer Lambda    

import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self,x):
        return self.func(x)
    
def preprocess(x):
    return x.view(-1, 1, 28, 28)
"""
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
"""

class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.AvgPool2d(4)
    def forward(self, x):
        
        x = x.view(-1, 1, 28, 28)#Input 维度：n_batch, n_channel, n_height, n_weight
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.fc1(x)
        
        return x.view(x.size(0), -1) # n_batch * n_class

def criterion(output, target):  
    return F.cross_entropy(output, target)
    
learning_rate = 0.1

model =  Mnist_CNN()
model.zero_grad()

opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

def fit(epochs, model, opt, train_dl, valid_dl, criterion):
    model.train()
    for epoch in range(epochs):
        
        correct_train=0
        total_train=0
        loss_train =0
        for ib, data in enumerate(train_dl):
            xb, yb = data
            pred = model(xb)
            
            loss = criterion(pred, yb)
            loss_train += loss
            loss.backward()
            
            opt.step()
            opt.zero_grad()
            
            
            
            correct_train += (torch.argmax(pred, 1) == yb).sum().item()
            total_train += xb.size(0)
            loss_valid = 0
            
            
            if ib % 1000 == 99:
                with torch.no_grad():
                    model.eval()
                    correct_valid = 0
                    total_valid = 0
                    
                    for xb_v, yb_v in valid_dl:
                        pred_v = model(xb_v)
                        correct_valid += (torch.argmax(pred_v, 1) == yb_v).sum().item()
                        total_valid += xb_v.size(0)
                        loss_valid += criterion(pred_v, yb_v)
                        
                        
                    print('--->epoch= %d iter= %d  Acc = %3f Loss = %3f on trainset and Acc = %3f Loss = %3f on validset' % 
                          (epoch, ib,  correct_train / total_train, loss_train / ib, 
                           correct_valid / total_valid, loss_valid / len(valid_dl)))


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")             

fit(5, model, opt, train_dl, valid_dl, criterion)


import matplotlib.pyplot as plt
import numpy as np
import torchvision


def imgshow(img):
    img = img*2 + 0.5 # unnormalize
    
    npimg = img.numpy()
    
    plt.imshow(np.transpose(npimg,(1,2, 0)))
    plt.show()
    
    
    
dataiter = iter(valid_dl)
img_iter, label_iter = dataiter.next()
imgshow(torchvision.utils.make_grid(img_iter))

