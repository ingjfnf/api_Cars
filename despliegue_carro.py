import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import sys
import os

# Definimos las columnas numéricas y categóricas como en el entrenamiento
categoricas = ['State', 'Make', 'Model']
numericas = ['Year', 'Mileage']

# Transformador para preprocesar los datos de entrada
# Este deberá ser idéntico al que se utilizó durante el entrenamiento
transformador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categoricas)
    ],
    remainder='drop'  # Descartar columnas que no sean numéricas ni categóricas
)

def predict_price(df_test, row_index):
    # Cargamos el modelo de predicción de precios
    clf = joblib.load(os.path.join(os.path.dirname(__file__), 'carros.pkl'))

    # Seleccionamos la fila que queremos predecir
    input_data = df_test.iloc[[row_index]]
    
    # Aplicamos la transformación al dato de entrada
    input_transformed = transformador.transform(input_data)

    # Hacemos la predicción
    precio_predicho = clf.predict(input_transformed)

    return precio_predicho

if __name__ == "__main__":
    # Carga del conjunto de datos de prueba
    df_test = pd.read_csv('dataTest_carListings.csv', index_col=0) # Asegúrate de tener el archivo en el directorio correcto

    if len(sys.argv) == 1:
        print('Por favor ingrese un número de ID para evaluar en el conjunto de TEST')
    else:
        row_index = int(sys.argv[1])
        p1 = predict_price(df_test, row_index)
        
        print(f'La predicción del precio del automóvil para el ID: {row_index} del conjunto de prueba es: {p1[0]}')

        
        
