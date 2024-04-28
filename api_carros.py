from flask import Flask
from flask_restx import Api, Resource, fields
from despliegue_carro import predict_price
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='1.0',
    title='API PREDICCIÓN DE CARROS GRUPO 14 MIAD',
    description='Esta es una API que utiliza un modelo de regresión lineal simple para predecir el precio de un carro, usted puede probar cuantas veces quiera y la cantidad de ID que desee'
)

ns = api.namespace('predict',
                   description='Predictor precio del carro')

parser = api.parser()
parser.add_argument(
    'ID',
    type=int,
    required=True,
    help='Introduzca el número del ID del conjunto de TEST que desea predecir con nuestro modelo',
    location='args'
)

resource_fields = api.model('Resource', {
    'Predicción': fields.Float,
})

@ns.route('/')
class CarPriceApi(Resource):
    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        row_index = args['ID']
        
        # Cargamos el conjunto de datos de prueba
        df_test = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)  

        try:
            prediction = predict_price(df_test, row_index)
            return {"Predicción": prediction[0]}, 200
        except Exception as e:
            api.abort(404, f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
