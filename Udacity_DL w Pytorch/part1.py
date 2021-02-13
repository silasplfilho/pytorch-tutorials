import torch

def activation(x):
    """
    Sigmoid activation function
    """
    return 1/(1+torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1, 5))

weights = torch.rand_like(features)

bias = torch.randn((1, 1))

y = activation(torch.sum(features * weights) + bias)
print(y)
y = activation((features * weights).sum() + bias)
print(y)

y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)

# --------------
# defining a state such that makes it reproducible
torch.manual_seed(7)

# Defining random values for the inputs
features = torch.randn((1, 3))

# Defining the structure of the NN
n_input = features.shape[1]  # number of cell inputs
n_hidden = 2  # n. of cell in hidden layer
n_output = 1  # n. of outputs

W1 = torch.randn(n_input, n_hidden)  # define weights which will be multiplied in 1st layer
W2 = torch.randn(n_hidden, n_output)  # define weights which will be multiplied in 2nd layer

B1 = torch.randn((1, n_hidden))  # define bias values to sum up after W1*inputs
B2 = torch.randn((1, n_output))  # define bias values to sum up after W2*W1

h1 = activation(torch.mm(features, W1) + B1)
y = activation(torch.mm(h1, W2) + B2)

y