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
    title='Car Price Prediction API',
    description='API to predict car prices'
)

ns = api.namespace('predict',
                   description='Car Price Predictor')

parser = api.parser()
parser.add_argument(
    'row_index',
    type=int,
    required=True,
    help='Row index of the test dataset to predict the car price',
    location='args'
)

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
        
        # Cargamos el conjunto de datos de prueba
        df_test = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)  # Asegúrate de tener el archivo en el directorio correcto

        try:
            prediction = predict_price(df_test, row_index)
            return {"result": prediction[0]}, 200
        except Exception as e:
            api.abort(404, f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
