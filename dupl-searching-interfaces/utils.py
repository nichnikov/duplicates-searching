import io
import requests
import pandas as pd
from flask import Response
from flask_restplus import inputs
from werkzeug.datastructures import Headers, FileStorage


# описание метода add_argument:
def api_configurator(name_space):
    """"""
    upload_parser = name_space.parser()
    upload_parser.add_argument("searched_texts", type=FileStorage, location='files', required=True,
                               help="искомые тексты, обязательные поля: id, texts")
    upload_parser.add_argument("texts_search_in", type=FileStorage, location='files', required=True,
                               help="тексты в которых искать, обязательные поля: id, texts")
    upload_parser.add_argument("only_different_groups", type=inputs.boolean, default=True, required=True,
                               help="True: возвращает похожие с разными id, False: возвращает похожие с любыми id")
    upload_parser.add_argument("score", type=float, required=True,
                               help="степень похожести")
    return upload_parser


def remote_duplicates_search(args, clustering_url, upload_type="excel"):
    """"""
    if upload_type == "json":
        json_data = args
    else:
        if upload_type == "excel":
            searched_texts_df = pd.read_excel(args['searched_texts'])
            texts_search_in_df = pd.read_excel(args['texts_search_in'])
        else:
            """The function expects csv type of upload data"""
            searched_texts_df = pd.read_csv(args['searched_texts'])
            texts_search_in_df = pd.read_csv(args['texts_search_in'])

        json_data = {"searched_texts": [{"text": tx, "id": id} for tx, id in zip(searched_texts_df["texts"],
                                                                                 searched_texts_df["id"])],
                     "texts_search_in": [{"text": tx, "id": id} for tx, id in zip(texts_search_in_df["texts"],
                                                                                  texts_search_in_df["id"])],
                     "only_different_groups": args['only_different_groups'],
                     "score": args['score']}

    duplicates_search_response = requests.post(clustering_url, json=json_data)
    duplicates_tuples_dict = duplicates_search_response.json()
    return pd.DataFrame(duplicates_tuples_dict["duplicates"], columns=["searched_text", "searched_id",
                                                                       "similar_text", "similar_text_id", "score"])


def response_func(clustering_texts_df, response_type="excel"):
    """"""
    headers = Headers()
    if response_type == "excel":
        headers.add('Content-Disposition', 'attachment', filename="searched_duplicates.xlsx")
        buffer = io.BytesIO()
        clustering_texts_df.to_excel(buffer, index=False, encoding='cp1251'),
        return Response(buffer.getvalue(),
                        mimetype='application/vnd.ms-excel',
                        headers=headers)

    else:
        headers.add('Content-Disposition', 'attachment', filename="searched_duplicates.csv")
        return Response(clustering_texts_df.to_csv(index=False),
                        mimetype="text/csv",
                        headers=headers)
