# Diving Deep into Supervised Training

import torch
import torch.nn as nn
import torch.optim as optim


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


input_dim = 2
lr = 0.001

perceptron = Perceptron(input_dim=input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)

for epoch_i in range(n_epochs):
    for batch_i in range(n_batches):

        x_data, y_target = get_toy_data(batch_size)

        perceptron.zero_grad()

        y_pred = perceptron(x_data, apply_sigmoid=True)

        loss = bce_loss(y_pred, y_target)

        loss.backward()

        optimizer.step()