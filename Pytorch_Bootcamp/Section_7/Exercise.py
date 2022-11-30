import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('Data/income.csv')


class TabularModel(nn.Module):
    """
    Modelo de rede neural que lida com dados tabulares. Os parametros de entrada são:
        emb_szs (lista de tuplas): atributos/features categóricas sao transformadas em embeddings - aqui passo a lista dessa transformacao
        n_count (int): numero de atributos/features continuos
        out_sz: quantidade de classes de saida - no caso de regressao, sera apenas um. No caso de uma classificacao binaria, 2
        layers (lista de ints): lista q contem a quantidade de neuronios em cada camada
        p (float): 
    """

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



## 1. Separate continuous, categorical and label columns
df.columns
df.describe()

cat_cols = ['age', 'sex', 'education', 'education-num', 'marital-status', 'workclass', 'occupation']
cont_cols = ['hours-per-week', 'income']
y_col = ['label']

## 2. Convert categorical columns to category dtypes
df[cat_cols].dtypes
for cat in cat_cols:
    df[cat] = df[cat].astype('category')  # transforma o tipo do atributo para categorico
df[cat_cols].dtypes

# SHUFFLE - THIS CELL IS OPTIONAL
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()

## 3. Set the embedding sizes
# Para criar o embeddings das variaveis categoricas, eu preciso preprocessar as variaveis categoricas

# transformando categorias em valores
age = df['age'].cat.codes.values
sex = df['sex'].cat.codes.values
education = df['education'].cat.codes.values
educ_number = df['education-num'].cat.codes.values
martl_status = df['marital-status'].cat.codes.values
workclass = df['workclass'].cat.codes.values
occupation = df['occupation'].cat.codes.values

# rearranjando as variaveis categoricas como uma matriz
cats = np.stack([age, sex, education, educ_number, martl_status, workclass, occupation], axis=1)
# cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)  # comando que realiza o mesmo processo de transformacao
# transformando np.array em um tensor
cats = torch.tensor(cats, dtype=torch.int64)
# usando embedding para representar one-hot-encoding dos atributos categoricos
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2 )) for size in cat_szs]


# rearranjando as variaveis continuas como uma matriz
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)

y = torch.tensor(df[y_col].values, dtype=torch.float)

##  Definindo otimizador e metrica para perda, batch_size e 
torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200, 100], p=0.4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(batch_size*0.2)

torch.manual_seed(33)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
