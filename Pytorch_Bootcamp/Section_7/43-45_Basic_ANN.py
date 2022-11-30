import torch 
import torch.nn as nn
import torch.nn.functional as F

# Definindo a arquitetura da rede neural  -  aula 43
class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):  # definindo a qtd de neuronios p cada layer
        # how many layers
        super().__init__()  # instancia o modulo herdado
        self.fc1 = nn.Linear(in_features, h1)  # full conected layer (linear)
        self.fc2 = nn.Linear(h1, h2)  
        self.out = nn.Linear(h2, out_features)  

        # input layer (4 features) --> h1 Number of neurons --> h2  M --> output (3 classes) 

    def forward(self, x):
        x = F.relu(self.fc1(x))  # funcao de ativacao  -  rectified linear units
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(32)
model = Model()

# aula 44
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/iris.csv')

X = df.drop('target', axis=1)
y = df['target']

X = X.values
y = y.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()  # escolhendo a metrica para a medicao do erro

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # lr = learning rate - menor o lr, mais lento sera o treinamento

# Definicao da qtd de epochs
epochs = 100
losses = []

for i in range(epochs):
    # executar treinamento nos dados e faer uma predicao
    y_pred = model.forward(X_train)

    # medir a perda/erro
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i%10==0:
        print(f'epoch:{i} and loss: {loss}')

    # backrpopagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

losses_detach = [ loss.detach().numpy() for loss in losses]
plt.plot(range(epochs),losses_detach)
plt.ylabel('Loss')
plt.xlabel('Epcoh')
plt.show()

## part three
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)

loss
correct=0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1}.)     {str(y_val)}    {y_test[i]}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'{correct} correct')

torch.save(model.state_dict(), 'Section 7/my_iris_model.pt')  # salva apenas os pesos e biases - posso retirar a opcao state_dict p salvar td
new_model = Model()
new_model.load_state_dict(torch.load('Section 7/my_iris_model.pt'))
new_model.eval()


# prever a classe para uma entrada aleatoria
mystery_iris =torch.tensor([5.6, 3.7, 2.2, 0.5])

with torch.no_grad():
    print(new_model(mystery_iris))
    print(new_model(mystery_iris).argmax())