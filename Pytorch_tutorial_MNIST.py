# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:37:11 2020

@author: zhang_2020
"""


from pathlib import Path
import requests

DATA_PATH = Path(r"D:\ZHANG2020\PytorchX\data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents = True, exist_ok = True)

#URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

#if not (PATH / FILENAME).exists():
#        content = requests.get(URL + FILENAME).content
#        (PATH / FILENAME).open("wb").write(content)


"""
This dataset is in numpy array format, and has been stored using pickle, 
a python-specific format for serializing data.
"""

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

"""
import matplotlib.pyplot as plt


plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
plt.show()
"""

import torch
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))

n, c = x_train.shape

from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size = 32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size = 32)


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3,stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3,stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size = 3,stride = 2, padding = 1)
        
    def forward(self, x):
        
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.avg_pool2d(x, 4)

        
        return x.view(-1, x.size(1))
        

import torch.optim as optim


dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)




#train_dl = WrappedDataLoader(train_dl, preprocess)
#valid_dl = WrappedDataLoader(valid_dl, preprocess)



net = Net()
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)

criterion = F.cross_entropy
loss_total = 0
net.zero_grad()

for epoch in range(10):
    for it,data in enumerate(train_dl):
        
        x_iter,y_iter = data
        y_pred = net(x_iter)
        loss = criterion(y_pred, y_iter)
        
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()
        
        if it % 500 == 0:
            #print('Iter = %d  loss = %3f' % (it, loss.item()))
            with torch.no_grad():
                correct = 0
                total = 0
                for xb, yb in valid_dl:
                    yp = net(xb)
                    correct += (torch.argmax(yp,1) == yb).sum().item()
                    total += yp.size(0)
                print('iter = %d  acc = %3f' % (it , correct/total))

PATH = r'D:\ZHANG2020\PytorchX\trained_net\mnist_cnn.pth'
torch.save(net.state_dict(), PATH)


F.normalize(train_dl, (0.5),(0.5))

