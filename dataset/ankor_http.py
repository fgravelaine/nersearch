import flask
from flask_restful import reqparse, abort, Api, Resource
from transformers import pipeline

app = flask.Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

model_checkpoint = "/Users/francois.gravelaine/.ankorsearchv1/models/hf/ankorsearchv1"
token_classifier = pipeline(
    "ner", model=model_checkpoint, aggregation_strategy="simple"
)
#tokens = token_classifier("Yellow towel Cruelty free Eco-Friendly organic made in France")

class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        print(args)

        tokens = token_classifier(user_query)
        end_tokens = []
        prev = None

        for token in tokens:
            if prev and token['start'] == prev['end']:
                if '##' in token['word']:
                    token['word'] = token['word'][2:]
                prev['word'] += token['word']
                prev['end'] = token['end']
            else:
                if prev:
                    end_tokens.append(prev)
                prev = token

        if prev:
            end_tokens.append(prev)

        # create JSON object
        output = end_tokens

        return str(output)


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)
