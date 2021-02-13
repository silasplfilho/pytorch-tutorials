import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Instatiation
# Creating tensor data
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# Creates a matrix
M_data = [[1., 2., 3.], [4. ,5. ,6.]]
M = torch.tensor(M_data)
print(M)

# Creates a 3d tensor of 2x2x2 dimension
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)

# random tensor
x = torch.randn((3, 4, 5))
print(x)

# Indexing
# Index into V
print(V[0])

# Index into M
print(M[0])

# Index into T
print(T[0])

# Operations
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)

# concatenation
# by rows
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# by columns
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1) # argument 1 indicates the axis
print(z_2)

# If your tensors are not compatible, torch will complain.  Uncomment to see the error
# torch.cat([x_1, x_2])

# Reshaping tensor objects
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
print(x.view(2, -1))

# Computation Graph
# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3.], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously could
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = x + y
print(z)
print(z.grad_fn)

# sum all entries in z
s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)

