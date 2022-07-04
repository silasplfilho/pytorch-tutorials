import torch

dd = [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]

t = torch.tensor(dd)
t.shape

t.reshape(1,9)
t.shape

ee = [[
        [2,3,4,4],
        [2,3,4,5]
    ]]

t = torch.tensor(ee)
t.shape

t.reshape(1,9)
t.shape

# VIDEO 8
import numpy as np

# usando numpy arrays
data = np.array([1, 2, 3])
type(data)

torch.from_numpy(data)

# sem usar numpy
torch.eye(2)  # cria uma especie de matriz identidade
torch.zeros(2,2,2)  # cria um tensor com elementos que sao apenas 0's
torch.ones(2,2)  # cria um tensor com elementos que sao apenas 1's
torch.rand(2,2)  # cria um tensor com escalares randomicos

# VIDEO 9
## MELHORES FORMAS DE CRIAR UM TORCH TENSOR

data = np.array([1, 2, 3])

t1 = torch.Tensor(data)  # Constructor function
t2 = torch.tensor(data)  # factory function
t3 = torch.as_tensor(data)  # factory function
t4 = torch.from_numpy(data)  # factory function

print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

# VIDEO 10
## OPERATIONS WITH TENSORS


# VIDEO 11
f