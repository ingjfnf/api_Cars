#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# In[2]:


# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)


# In[3]:


# Definimos las columnas numéricas y categóricas
categoricas = ['State', 'Make', 'Model']
numericas = ['Year', 'Mileage']

# Con la herramienta ColumnTransformer de scikit-learn vamos a realizarle el one hot encoding a las columnas categoricas y además vamos a estandarizar los valores de las columnas numericas
transformador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas)
    ])


# In[4]:


X = dataTraining.drop('Price', axis=1)
y = dataTraining['Price']


# In[5]:


# Ajustamos y transformamos los datos de entrenamiento.
X_ajustadas = transformador.fit_transform(X)


# In[6]:


# Creamos el modelo de regresión lineal
model = LinearRegression()

# Entrenamos el modelo
model.fit(X_ajustadas, y)


# In[8]:


# Exportar modelo a archivo binario .pkl
joblib.dump(model,  'carros.pkl', compress=3)

joblib.dump(model,  './carros.joblib')

joblib.dump(transformador, 'transformador.pkl')


# In[ ]:




