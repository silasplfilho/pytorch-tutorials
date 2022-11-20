from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
        input_dim (int): the size of the input vectors
        hidden_dim (int): the output size of the first Linear layer
        output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP
        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output


batch_size = 2
input_dim = 3
hidden_dim = 100
output_dim = 4

mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)
print(mlp)

def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

# Example 4-3. Testing the MLP with random inputs
x_input = torch.rand(batch_size, input_dim)
describe(x_input)

# Example 4-4. Producing probabilistic outputs with a multilayer perceptron classifier (notice the apply_softmax = True option)
y_output = mlp(x_input, apply_softmax=True)
describe(y_output)