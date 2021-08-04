import torch


X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()


learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()  # calculate the gradient of loss of w (dl/dw)

    with torch.no_grad():
        w -= learning_rate * dw

    # zero gradients
    w.grad.zero_()