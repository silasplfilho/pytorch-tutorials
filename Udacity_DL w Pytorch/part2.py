import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from math import exp

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

# =============
# Constructing the network to deal with MNIST dataset
def activationFunction(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


inputs = images.view(images.shape[0], -1)  # nivela o batch definido

n_input_layer = 784
n_hidden_layer = 256
n_output_layer = 10

W1 = torch.randn(n_input_layer, n_hidden_layer)  # define weights which will be multiplied in 1st layer
W2 = torch.randn(n_hidden_layer, n_output_layer)  # define weights which will be multiplied in 2nd layer

B1 = torch.randn((1, n_hidden_layer))  # define tensor w/ bias values to sum up after W1*inputs
B2 = torch.randn((1, n_output_layer))  # define tensor w/ bias values to sum up after W2*W1

h = activationFunction(torch.mm(inputs, W1) + B1)
y = activationFunction(torch.mm(h, W2) + B2)
print(y.sum(dim=1))

# =============
# Using pytorch definitions of layers

from torch import nn

# class Network(nn.Module):
#     def __init__(self):  # defining constructor
#         super().__init__()

#         # Defining layers
#         self.hidden = nn.Linear(784, 256)  # hidden layer
#         self.output = nn.Linear(256, 10)  # output layer

#         # Defining activation functions
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # Defining the traffic of the network
#         x = self.hidden(x)
#         x = self.sigmoid(x)
#         x = self.output(x)
#         x = self.softmax(x)

#         return x

# model = Network()
# model

import torch.nn.functional as F


# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = nn.Linear(784, 256)
#         self.output = nn.Linear(256, 10)

#     def forward(self, x):
#         x = F.sigmoid(self.hidden(x))
#         x = F.softmax(self.output(x), dim=1)

#         return x

# =============
# Exercise: implement a network with 2 hidden ayers, each one with different activation functions

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim=1)

        return x

model = Network()

# Printing hidden layer 1 bias and weights
print(model.hidden1.weight)
print(model.hidden1.bias)

# Set biases to all zeros
model.hidden1.bias.data.fill_(0)
# sample from random normal with standard dev = 0.01
model.hidden1.weight.data.normal_(std=0.01)

## Forward pass 
# Grab some data 
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels) 
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size

# Forward pass through the network
img_idx = 0
ps = model.forward(images[img_idx,:])

img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)