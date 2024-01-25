
from pathlib import Path
import torch
from torch import nn  # contem as diferentes arquiteturas de Redes Neurais
import matplotlib.pyplot as plt

torch.__version__

# Data Preparing and Loading
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# 1. split em treino e teste
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

## Visualizacao da distribuicao dos dados 
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c='b', s=4, label="training data")

    plt.scatter(test_data, test_labels, c='g', s=4, label="testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label="predictions")

    plt.legend(prop={"size": 14})
    plt.show()

plot_predictions()

# 2. Build Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
        
torch.manual_seed(42)
model_0 = LinearRegressionModel() 
list(model_0.parameters())

model_0.state_dict()

### fazendo predicoes
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

# 3. Treino do modelo

## escolhendo funcao perda
loss_fn = nn.L1Loss()

## Escolhendo otimizador
optim = torch.optim.SGD(params=model_0.parameters(),
                        lr=0.01)

# 
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

###Training
for epoch in range(epochs):
    model_0.train()

    # 1. forward
    y_pred = model_0(X_train)

    # 2. calcular perda
    loss = loss_fn(y_pred, y_train)
    print(f"Loss: {loss}" )

    # 3. Otimizar gradiente descend
    optim.zero_grad()

    # 4. Executar backpropagation
    loss.backward()

    # 5. Definir otimizador por meio gradiente descend
    optim.step()

    # Testing 
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward Pass
        test_pred = model_0(X_test)
        # 2. calcular perda
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Test: {loss} | Test Loss: {test_loss}")
        model_0.state_dict()
        
    

plot_predictions(predictions=test_pred)

plt.plot(epoch_count, torch.tensor(loss_values).detach().numpy(), label="Train Loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Salvando o modelo
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

# Criando um modelo novo e Carregando os parametros treinados
loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.state_dict()

loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)