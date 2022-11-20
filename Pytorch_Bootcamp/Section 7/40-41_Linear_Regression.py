import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

## 1o fazendo uma regressao linear simples usando dados aleatorios

# Definindo dados aleatorios que serão usados na regressao linear
X = torch.linspace(1,50,50).reshape(-1,1)
print(X)

# Definindo um array de valores de erros aleatorios para o modelo
torch.manual_seed(71) # to obtain reproducible results
e = torch.randint(-8,9,(50,1),dtype=torch.float)
print(e.sum())

# Defininfo a funcao y
y = 2*X + 1 + e
print(y.shape)

# definindo modelo de regressao linear
model = nn.Linear(in_features=1, out_features=1)

# plotando os dados da funcao
plt.scatter(X.numpy(), y.numpy())
plt.ylabel('y')
plt.xlabel('x')
plt.show()


# definicao da classe que instancia o modelo de rede neural utilizada
class Model(nn.Module):  # utiliza herda biblioteca nn do pytorch, que possui implementacoes de arquiteturas de ANN

    def __init__(self, in_features, out_features):  #metodo construtorr, que instancia as variaveis toda vez q há uma instanciacao da classe
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

torch.manual_seed(59)

model = Model(1, 1)

print(model.linear.weight)
print(model.linear.bias )


