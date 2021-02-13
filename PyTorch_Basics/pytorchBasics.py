import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

# Defining a tensor in pytorch
describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3))
describe(torch.randn(2, 3))

describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
describe(x)
x.fill_(5)

x = torch.Tensor([[1, 2, 3], 
                [4, 5, 6]])
describe(x)                

x.long()

# Tensor operations
describe(torch.add(x, x))
describe(x + x)

## Operation on dimension
x = torch.arange(6)
describe(x)
x = x.view(2, 3)
describe(x)

describe(torch.sum(x, dim=0))s

