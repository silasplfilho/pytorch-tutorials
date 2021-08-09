import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]  # n_samples, 1
        self.n_samples = xy.shape[0]
        # data loading

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
firstData = dataset[0]
features, labels = firstData
print(features)
print(type(features), type(labels))


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)

firstData = dataset[0]
features, labels = firstData
print(features)
print(type(features), type(labels))

# dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# # dummy training loop
# num_epochs = 2
# total_samples = len(dataset)
# n_iterations = math.ceil(total_samples / 4)
# print(total_samples, n_iterations)

# for epoch in range(num_epochs):
#     for i, (inputs, labels) in enumerate(dataloader):
#         # forward, backward, update
#         if (i+1) % 5 == 0:
#             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# # torchvision.datasets.MNIST()
# # fashion-mnist, cifar, coco
