#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_price(fila):
    # Cargamos el modelo de predicción de precios
    clf = joblib.load(os.path.dirname(__file__) + '/carros.pkl') 

    # Hacemos la predicción
    precio_predicho = clf.predict(fila)

    return precio_predicho

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Por favor ingrese un número de ID para evaluar en el conjunto de TEST')
    else:
        fila_t = sys.argv[1]
        p1 = predict_price(fila_t)
        
        print(fila_t)
        print('La predicción del precio del automóvil para el ID:', sys.argv[1], 'del conjunto de prueba es:', p1)

        
        