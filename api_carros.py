from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from despliegue_carro import predict_price  
from flask_cors import CORS
import pandas as pd  

app = Flask(__name__)
CORS(app)  

api = Api(
    app, 
    version='1.0', 
    title='Car Price Prediction API',
    description='API to predict car prices')

ns = api.namespace('predict', 
     description='Car Price Predictor')
   
parser = api.parser()

parser.add_argument(
    'row_index', 
    type=int, 
    required=True, 
    help='Row index of the test dataset to predict the car price', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.Float,
})

@ns.route('/')
class CarPriceApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        row_index = args['row_index']
        
        try:
            # Asegúrate de que tu DataFrame 'df_test' es el correcto y está cargado
            test_row = df_test.iloc[[row_index]].values.reshape(1, -1)  # Asumiendo que el modelo espera un array 2D
            prediction = predict_price(test_row)  # Usamos la función correcta 'predict_price'
            return {"result": prediction[0]}, 200  # Devolvemos el valor predicho, asumiendo que 'predict' devuelve un array
        except Exception as e:
            api.abort(404, f"Error: {str(e)}")

if __name__ == '__main__':
    # Cargamos el conjunto de datos de prueba desde una URL
    df_test = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
