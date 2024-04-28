#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_price(df_test, row_index):
    # Cargamos el modelo de predicción de precios y el transformador
    clf = joblib.load(os.path.join(os.path.dirname(__file__), 'carros.pkl'))
    transformador = joblib.load(os.path.join(os.path.dirname(__file__), 'transformador.pkl'))
    
    # Seleccionamos la fila que queremos predecir
    input_data = df_test.iloc[[row_index]]
    
    # Aplicamos la transformación al dato de entrada usando el transformador 
    input_transformed = transformador.transform(input_data)

    # Hacemos la predicción
    precio_predicho = clf.predict(input_transformed)

    return precio_predicho

if __name__ == "__main__":

    df_test = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)  # Cambia la ruta si el archivo está en otro lugar

    if len(sys.argv) == 1:
        print('Por favor ingrese un número de ID para evaluar en el conjunto de TEST')
    else:
        row_index = int(sys.argv[1])
        p1 = predict_price(df_test, row_index)
        
        print(f'La predicción del precio del automóvil para el ID: {row_index} del conjunto de prueba es: {p1[0]}')


        
        
