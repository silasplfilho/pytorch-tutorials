import torch

# import sympy as sym
x_data = [1., 2., 3.]
y_data = [2., 4., 6.]
w = torch.tensor([1.], requires_grad=True)
# w = 1.  # peso que sera usado com os valores de x_data

# w_1 = 1.  # peso para exercicio
# w_2 = 1.  # peso para exercicio


# modelo definido
def forward(x):
    return x * w
    # return ((x * x) * w_2) + (x * w_1)  # para exercicio


# funcao de perda
def loss(y_pred, y_val):
    # y_pred = forward(x)
    return (y_pred - y_val) ** 2


# # gradiente
# def gradient(x, y):  # d_loss/d_w
#     return 2 * x * (x * w - y)
    # return 2 * ((2 * w_2 * x) + w_1) * ((w_2 * x**2) + (w_1 * x) - y)  # p/ exercicio


# def derivative():
#     x, y, w = sym.Symbol('x y w')
#     f = (w*x - y) ** 2
#     return sym.diff(f, x)

#  Antes treinamento
print("Valor predito (antes do treinamento)", 4, forward(4).item())

# Loop de Treinamento
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)  # forward pass
        l = loss(y_pred, y_val)  # compute loss
        l.backward()  # backpropagation to update the weights

        print("\tgrad: ", x_val, y_val, grad)
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updtae weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")

# Apos treinamento
print("predict (after training)", "4 hours", forward(4))
