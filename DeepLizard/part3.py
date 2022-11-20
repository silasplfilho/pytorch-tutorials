from numpy import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# -----------------
# DEFININDO A ARQUITETURA DA REDE NEURAL
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)


    def forward(self, t):
        # (2) hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)  # 12 é o tamanho do output da camada anterior, 4 * 4 é a dimensao da img, de cada uma das 12 saidas da camada anterior
        t = F.relu(self.fc1(t))

        # (5) hidden linear layer
        t = F.relu(self.fc2(t))

        # (6) output layer
        t = F.softmax(self.out(t))

        return t

# -----------------
# INSTANCIANDO A CNN
network = Network()

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(False)


# ---
# DEFININDO O DATASET E DEIXANDO NO FORMATO DE TENSOR
train_set = torchvision.datasets.FashionMNIST(
    root = './DeepLizard/data/FashionMNIST',
    train = True,
    download = True,  # Extract step
    transform=transforms.Compose([transforms.ToTensor()])  # Transform step
)

sample = next(iter(train_set))  # OBJETO DO TIPO DATASET PERMITE ITERACAO
image, label = sample

image.unsqueeze(0).shape  # gives a batch with size 1

pred = network(image.unsqueeze(0))  # PRECISA DEIXAR O INPUT NO FORMATO APROPRIADO, NO CASO EM TENSOR DE TAMANHO 1

# ----
data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10
)

batch = next(iter(data_loader))
images, labels = batch

preds = network(images)
preds.argmax(dim=1)
preds.argmax(dim=1).eq(labels)
preds.argmax(dim=1).eq(labels).sum()
