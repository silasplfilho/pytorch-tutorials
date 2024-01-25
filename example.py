# Codigo do tutorial: https://github.com/ajhalthor/deep-learning-101/tree/main

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset

# data import
iris = load_iris()
X = iris.data
y = iris.target

X.shape, y.shape

# divisao em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train[0], y_train[0]

# Convertendo numpy array em tensores pytorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

X_train_tensor[0], y_train_tensor[0]

type(X_train), type(y_train), type(X_train_tensor), type(y_train_tensor)

batch_size = 5
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Definicao da arquitetura da rede neural
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hiperparametros
input_size = 4
hidden_size = 6
num_classes = 3
learning_rate = 0.001
num_epochs = 1_000

model = NeuralNetwork(input_size, hidden_size, num_classes)
model

# Funcao Perda
criterion = nn.CrossEntropyLoss()
criterion

# Otimizador
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer

# Treinando o modelo
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # passo forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # passo p/ tras e otimizacao
        optimizer.zero_grad()  # limpa gradientes
        loss.backward()  # computa gradientes
        optimizer.step()  #atualiza parametros da NN

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}')
        print(batch_X.shape, batch_y.shape, loss)

# Testando o modelo
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)  # batch_size x 3
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = correct/total
    print(f'Test Accuracy: {accuracy:.2f}')