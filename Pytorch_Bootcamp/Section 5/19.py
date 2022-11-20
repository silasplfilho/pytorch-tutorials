import torch
import numpy as np

## Convertendo numpy arrays para tensores
arr = np.array([1, 2, 3, 4, 5])
print(arr)

x = torch.from_numpy(arr)  # converte um array numpy para um tensor - mantendo um link entre os objetos criados
type(x)

x1 = torch.as_tensor(arr)  # converte um array numpy para um tensor - mantendo um link entre os objetos criados
type(x1)

x1.dtype

# 
arr2d = np.arange(0.0, 12.0)
arr2d = arr2d.reshape(4, 3)

x2 = torch.from_numpy(arr2d)  # caso eu altere o objeto de referencia, o elemento dentro do tensor tambem sera afetado 
arr2d[0,1] = 100
# 
arr[0] = 99
x1

x3 = torch.tensor(arr)
arr[1] = 88

## Criando tensores
torch.empty(2, 2)

torch.zeros(2, 3, dtype=torch.int64)

torch.ones(3, 2)

# tensor dentro de um range
x = torch.arange(0, 18, 2).reshape(3, 3)

# trocando o tipo do tensor
x.type(torch.int64)

# tensor coim valores aleatorios
torch.rand(4, 3)

# tensor com valores aleatorios dentro de uma normal
torch.randn(4, 3)

# tensor com valores inteiros
torch.randint(low=0, high=10, size=(5,5))

