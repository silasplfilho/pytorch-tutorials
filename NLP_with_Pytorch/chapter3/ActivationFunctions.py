from turtle import forward
import torch
import torch.nn as nn


class Perceptron(nn.Module):
    """A perceptron is one linear layer"""
    
    def __init__(self, input_dim):
        """ 
        Args: input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron
        Args: 
            x_in (torch.Tensor): an input data tensor 
            x_in.shape should be (batch, num_features)
        Returns: the resulting tensor. tensor.shape should be (batch,).
        """

        return torch.sigmoid(self.fc1(x_in)).squeeze()


# ================================
# Activation Functions Examples
# Example 3-2
import matplotlib.pyplot as plt

x = torch.range(-5., 5., 0.1)
y = torch.sigmoid(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3-3
x = torch.range(-5., 5., 0.1)
y = torch.tanh(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3-4
relu = t orch.nn.ReLU()
x = torch.range(-5., 5., 0.1)
y = relu(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3-5
prelu = torch.nn.PReLU(num_parameters=1)
x = torch.range(-5., 5., 0.1)
y = prelu(x)

plt.plot(x.numpy(), y.detach().numpy())
plt.show()


# Example 3-6
softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)

print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))
