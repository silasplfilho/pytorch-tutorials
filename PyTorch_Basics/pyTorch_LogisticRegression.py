# Tutorial from http://bit.ly/PyTorchVideo
import torch.nn.functional as F
import torch
###################################################################################################
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])


# 1 - Definicao do modelo utilizado
class Model(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = F.sigmoid(self.linear(x))  # funcao de ativacao Sigmoid
        # y_pred = F.relu6(self.linear(x))  # funcao de ativacao ReLU
        return y_pred


model = Model()
###################################################################################################
#  2 - Definicao da funcao de perda e otimizacao
#  loss & optimizer
criterion = torch.nn.BCELoss(reduction='mean')  # Entropia Cruzada Binaria
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # gradiente descendente estocastico


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

# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(torch.tensor([[1.0]]))
print(
    f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(torch.tensor([[7.0]]))
print(
    f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
