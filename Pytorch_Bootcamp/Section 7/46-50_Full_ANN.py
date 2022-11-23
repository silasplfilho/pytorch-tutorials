import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/NYCTaxiFares.csv')

df.describe()
df.columns
df['fare_amount'].describe()


## feature engineering (criacao de novas features a partir das ja existentes)- aula 47 - Part 1

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    funcao de haversine - mede a distancia entre dois pontos usando latitude e longitude
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d

df["dist_km"] = haversine_distance(df, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude")

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["EDdate"] = df["pickup_datetime"] - pd.Timedelta(hours=4)
df["Hour"] = df["EDdate"].dt.hour
df["AMorPM"] = np.where(df['Hour']<12, 'am', 'pm')
df['Weekday'] = df['EDdate'].dt.strftime("%a")

df.head()

my_time = df["pickup_datetime"][0]
my_time

##  Separando variaveis categoricas e numericas - Aula 48 - Part 2

# variavel com nome das features categoricas
cat_cols = ['Hour', 'AMorPM', 'Weekday']

# variavel com nome das features continuas
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']

y_col = ['fare_amount']

df.dtypes
for cat in cat_cols:
    df[cat] = df[cat].astype('category')  # transforma o tipo do atributo para categorico

df.dtypes

# transformando categorias em valores
hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

# rearranjando as variaveis como uma matriz
cats = np.stack([hr, ampm, wkdy], axis=1)
# cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)  # comando que realiza o mesmo processo de transformacao

# transformando np.array em unm tensor
cats = torch.tensor(cats, dtype=torch.int64)

conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y_col].values, dtype=torch.float)

cats.shape
conts.shape

# usando embedding para representar one-hot-encoding dos atributos categoricos
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2 )) for size in cat_szs]


## Implementando Regressão - Aula 49 - Parte 3

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum([nf for ni, nf in emb_szs])
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i

        layerlist.append(nn.Linear(layers[-1], out_sz))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, 1]))

        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x


## Treinamento e Validaçã0 - Aula 50
torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200, 100], p=0.4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(batch_size*0.2)

cat_train =cats[:batch_size-test_size] 
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size] 
con_test = conts[batch_size-test_size:batch_size]

y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

len(cat_train)


# Treinamento do modelo
import time
start_time = time.time()

epochs = 150
losses = []

for i in range(epochs):
    i += 1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train))
    losses.append(loss)

    if i%25 == 1:
        print(f'epoch: {i} loss is {loss}')


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

duration = time.time() - start_time
print(f"Training took {duration/60} minutes")

# plot the losses
losses_detach = [loss.detach().numpy() for loss in losses]
losses_detach[0]


plt.plot(range(epochs), losses_detach)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')
plt.show()


# TO EVALUATE THE ENTIRE TEST SET
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')

# Make sure to save the model only after the training has happened!
if len(losses) == epochs:
    torch.save(model.state_dict(), 'TaxiFareRegrModel.pt')
else:
    print('Model has not been trained. Consider loading a trained model instead.')