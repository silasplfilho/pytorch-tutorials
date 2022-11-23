import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/NYCTaxiFares.csv')

df.describe()
df.columns
df['fare_amount'].describe()


## feature engineering (criacao de novas features a partir das ja existentes)- aula 47

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

##  Separando variaveis categoricas e numericas - Aula 48

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