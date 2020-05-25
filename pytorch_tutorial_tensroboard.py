# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:01:52 2020

@author: zhang_2020
"""

import torch 
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])

trainset = torchvision.datasets.FashionMNIST("./data", 
                                             download=True,
                                             train=True,
                                             transform=transform)
testset = torchvision.datasets.FashionMNIST("./data",
                                            download=True,
                                            train=False,
                                            transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4)


# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
"""        
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self,x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
      
FMNIST_CNN = nn.Sequential(Lambda(preprocess),
                           nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                           nn.ReLU(),
                           nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
                           nn.ReLU(),
                           nn.AvgPool2d(4),
                           Lambda( lambda x : x.view(x.size(0),-1))
                           )


def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dl = WrappedDataLoader(trainloader, preprocess)
test_dl = WrappedDataLoader(testloader, preprocess)

FMNIST_CNN.to(dev)

"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9) 
loss_fn = F.cross_entropy

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fashion_mnist_experiment_1')


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)


"""
def fit(epochs, model, opt, loss_fn, train_dl):
    
    train_loss = 0
    i_count = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            
            pred = FMNIST_CNN(xb)
            
            loss = loss_fn(pred, yb)
            
            loss.backward()
            
            opt.step()
            opt.zero_grad()
        
            train_loss += loss.item()
            
            i_count+=1
            
            if i_count % 1000 ==0:
                print("train_loss = ",train_loss / i_count)
        print("train_loss = ",train_loss / len(train_dl))
        
        
def accuracy(model, test_dl):
    with torch.no_grad():
        correct = 0 
        for xb, yb in test_dl:
            pred = model(xb)
            
            correct += (torch.argmax(pred,1) == yb).sum().item()
            
            
        return correct, correct / len(test_dl) / 4
        

fit(1, FMNIST_CNN, optimizer, loss_fn, train_dl)

"""
