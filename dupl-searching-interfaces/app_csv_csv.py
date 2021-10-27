import os
from dotenv import load_dotenv
from flask import Flask
from flask_restplus import Api, Resource
from utils import remote_duplicates_search, response_func, api_configurator


load_dotenv(".env")
CLUSTERING_URL = os.environ.get("CLUSTERING_URL")
if CLUSTERING_URL is None: raise Exception('Env var CLUSTERING_URL not defined')

app = Flask(__name__)
api = Api(app)

csv_name_space = api.namespace('api', 'загрузка двух файлов csv с текстами (обязательные поля "texts" и "id" должны '
                                  'быть в каждом файле) и score (число)')

upload_parser = api_configurator(csv_name_space)


@csv_name_space.route('/csv_csv')
@csv_name_space.expect(upload_parser)
class ClusteringCsvCsv(Resource):
    def post(self):
        """POST method on input csv file with texts and score, output clustering texts  as csv file."""
        args = upload_parser.parse_args()
        clustering_texts_df = remote_duplicates_search(args, CLUSTERING_URL, upload_type="csv")
        return response_func(clustering_texts_df, response_type="csv")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6003)
