# VIDEO 15 - ETL process in pytorch
from turtle import forward
import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root = './DeepLizard/data/FashionMNIST',
    train = True,
    download = True,  # Extract step
    transform=transforms.Compose([transforms.ToTensor()])  # Transform step
)

## 
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10,
    shuffle=True)  # Load step

# VIDEO 16 
import numpy as np
import matplotlib.pyplot as plt

len(train_set)

train_set.targets  # targets sao as classes representadas por numeros # substitui o atributo train_labels

train_set.targets.bincount()  # conta a qtd de registros para cada valor de classe

sample = next(iter(train_set))
len(sample)
type(sample)
image, label = sample

image.shape
plt.imshow(image.squeeze(), cmap='gray')
plt.show()
print('label: ', label)

batch = next(iter(train_loader))
len(batch)
type(batch)
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
plt.show()


# VIDEO 17
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer = None
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

 

    def forward(self, t):
        return t


network = Network()
print(network)

network.conv1.weight