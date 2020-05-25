# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:01:22 2020

@author: zhang_2020
"""
# Tutorial-60min start Blitz
# 3rd-Neural Network 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super (Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        
        x = x.view(-1, self.num_flat_features(x))
    
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
net = Net()
params = list(net.parameters())

#print(len(params))
#print(params[0].size())  # conv1's .weight


###########################################################################
########################## forward propgation  ############################
###########################################################################
input = torch.randn(1, 1, 32, 32)
out =net(input)
#print(out)


#Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))


###########################################################################
########################## Compute the Loss ###############################
###########################################################################
output = net(input)

criterion = nn.MSELoss()
target = torch.randn(10).view(1,-1)

loss = criterion(output, target)
print('loss = ', loss)

#print(loss.grad_fn)  # MSELoss
#print(loss.grad_fn.next_functions[0][0])  # Linear
#print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


###########################################################################
########################## backward propgation    #########################
###########################################################################
net.zero_grad()
#print('grad before backward',net.conv1.bias.grad)

loss.backward()
#print('grad after backward',net.conv1.bias.grad)



###########################################################################
########################## update weights #################################
###########################################################################
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)

optimizer.zero_grad()
optimizer.step()


###########################################################################
########################## Get Started ## #################################
###########################################################################

