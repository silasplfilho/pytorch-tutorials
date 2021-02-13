# Tutorial from http://bit.ly/PyTorchVideo
from torch import nn, optim, from_numpy
import numpy as np
###################################################################################################
xy = np.loadtxt('PyTorch_Basics/data/diabetes.csv.gz', delimiter=',', dtype=np.float32)

x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, -1])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)  # 8 é a quantidade de colunas de x_data
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))

        return y_pred


model = Model()
###################################################################################################
#  2 - Definicao da funcao de perda e otimizacao
#  loss & optimizer
criterion = nn.BCELoss(reduction='mean')  # Entropia Cruzada Binaria
optimizer = optim.SGD(model.parameters(), lr=0.01)  # gradiente descendente estocastico

###################################################################################################
#  3 - Execucao e atualizacao dos pesos para cada iteracao
for epoch in range(1000):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

###################################################################################################
