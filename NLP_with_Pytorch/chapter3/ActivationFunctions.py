from turtle import forward
import torch
import torch.nn as nn

# ================================
# Section 1

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
# Section 2
# Exemplos de Funções de Ativação

# Example 3.2 - Sigmoid
import matplotlib.pyplot as plt

x = torch.range(-5., 5., 0.1)
y = torch.sigmoid(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3.3 - Tanh
x = torch.range(-5., 5., 0.1)
y = torch.tanh(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3.4 - Rectified Linear Unit
relu = torch.nn.ReLU()
x = torch.range(-5., 5., 0.1)
y = relu(x)

plt.plot(x.numpy(), y.numpy())
plt.show()


# Example 3.5 - Parametric Rectified Linear Unit
prelu = torch.nn.PReLU(num_parameters=1)
x = torch.range(-5., 5., 0.1)
y = prelu(x)

plt.plot(x.numpy(), y.detach().numpy())
plt.show()


# Example 3.6 - Softmax
softmax = nn.Softmax(dim=1)
x_input = torch.randn(1, 3)
y_output = softmax(x_input)

print(x_input)
print(y_output)
print(torch.sum(y_output, dim=1))


# ================================
# Section 3
# Exemplos de Funções de Perda

# Mean Square Error
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
print(loss)

# Categorical Cross-Entropy Loss
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
print(loss)

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
loss = bce_loss(probabilities, targets)
print(probabilities)
print(loss)
