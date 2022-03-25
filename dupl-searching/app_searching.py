from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields
from utils import duplicates_search_func

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
api = Api(app)

name_space = api.namespace('api', 'На вход поступает JSON, возвращает JSON')
one_item = name_space.model("One Item",
                            {"text": fields.String(description="Searched texts", required=True),
                             "id": fields.String(description="Texts in which to search", required=True),
                             })

input_data = name_space.model("Input JSONs",
                              {"score": fields.Float(description="The similarity coefficient", required=True),
                               "only_different_groups":
                                   fields.Boolean(description="If it is not necessary to search in the same group",
                                                  required=True),
                               "searched_texts": fields.List(fields.Nested(one_item)),
                               "texts_search_in": fields.List(fields.Nested(one_item))})


@name_space.route('/searching')
class Searching(Resource):
    """Service searches duplicates in two texts collections."""

    @name_space.expect(input_data)
    def post(self):
        """POST method on input JSON file with scores and two lists of dictionaries
        each of them contents texts and text's ids, output list with dictionaries with similarity texts and them
        similarity score as a JSON file."""
        json_data = request.json

        min_score = json_data["score"]
        only_different_groups = json_data["only_different_groups"]
        searched_texts = [d["text"] for d in json_data["searched_texts"]]
        searched_ids = [d["id"] for d in json_data["searched_texts"]]
        texts = [d["text"] for d in json_data["texts_search_in"]]
        ids = [d["id"] for d in json_data["texts_search_in"]]

        assert len(texts) == len(ids), "len of texts not equals len of ids"
        assert len(searched_texts) == len(searched_ids), "len of searched texts not equals len of searched ids"

        # ("searched_text, searched_id, similar_text, similar_text_id, score")
        search_results = duplicates_search_func(searched_ids, searched_texts, ids, texts, min_score)
        if only_different_groups:
            search_results = [x for x in search_results if x[1] != x[3]]
        return jsonify({"duplicates": search_results})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6004)
