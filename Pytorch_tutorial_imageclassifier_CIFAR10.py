# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:07:33 2020

@author: zhang_2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

"""
Specifically for vision, we have created a package called torchvision, that has 
data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data 
transformers for images, viz., torchvision.datasets and torch.utils.data.DataLoader.

"""

import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]. .. note:


# compose-> run codes in []
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(r'D:\\ZHANG2020\\PytorchX\\data', train = True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers=0)

testset = torchvision.datasets.CIFAR10(r'D:\\ZHANG2020\\PytorchX\\data', train=False, 
                                       download = True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    
    img = img*2 + 0.5 # unnormalize
    
    npimg = img.numpy()
    
    plt.imshow(np.transpose(npimg,(1,2, 0)))
    plt.show()
    
    
"""
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
"""

win_len    = 3

c1_out_len = 6
c2_out_len = 16
img_weight = 32
img_hidth = 32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        

        self.fc1 = nn.Linear(16 * 5 * 5), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        #print('before conv1: ',x.size())
        x = self.conv1(x)
        #print('after conv1: ',x.size())
        x = F.relu(x)
        #print('after relu: ', x.size())
        x = self.pool(x)
        #print('after pool: ',x.size())

        x = self.conv2(x)
        #print('after conv2: ',x.size())
        x = F.relu(x)
        #print('after relu: ', x.size())
        x = self.pool(x)
        #print('after pool: ',x.size())


        x = x.view(-1, 16 * 6 * 6 )
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        
        return x


net = Net()
net.zero_grad()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)

running_loss = 0.0

for epoch in range(2):# loop over the dataset multiple times
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        
        
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d,  %5d] loss: %3f' % ((epoch + 1, i + 1, running_loss / 2000)))
            
            running_loss = 0.0
        
print('Finished Training')        

PATH = r'D:\ZHANG2020\PytorchX\trained_net\cifar_net_10midlayers.pth'
torch.save(net.state_dict(), PATH)

"""

"""
# Test the net on the testset

dataiter =  iter(testloader)
images_test, labels_test = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


outputs_test = net(images_test)
_, predicted = torch.max(outputs_test, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))



""" 
# load trained model
PATH = r'D:\ZHANG2020\PytorchX\trained_net\cifar_net.pth'
net_re = Net()
net_re.load_state_dict(torch.load(PATH))
"""

import time


a = time.time()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data#[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
        
print(time.time()-a)


#device = torch.device("cpu")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

#net_re.to(device)

for params in net.parameters():
    print(params.data.size())