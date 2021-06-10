from gensim.models import KeyedVectors
from anki_corpus_for_gensim import bg_stopwords,en_stopwords

import json, argparse, time

from flask import Flask, request
from flask_cors import CORS

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)

def get_wmdist_bg(bg_sent, en_sent):
    return multilingual_model.wmdistance(
        ['bg:'+x for x in bg_sent if x not in bg_stopwords],
        ['en:'+x for x in en_sent if x not in en_stopwords]
    )

@app.route('/', methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    results = list()

    if data == "":
        params = request.form
        sentences = json.loads(params)
        print(sentences)
        # bg_sent = json.loads(params['bg']).split()
        # en_sent = json.loads(params['en']).split()
    else:
        params = json.loads(data)
        sentences = params
        print(sentences)
        # bg_sent = params['bg'].split()
        # en_sent = params['en'].split()

    for sent in sentences['sent']:
        print(sent)

        bg_sent = sent['bg'].split()
        en_sent = sent['en'].split()

        print(bg_sent)
        print(en_sent)

        match = get_wmdist_bg(bg_sent, en_sent)
        results.append(match)

    print('Returning', results)

    json_data = json.dumps({'results': results})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data


##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    multilingual_model = KeyedVectors.load('glove_word_embeddings/multilingual.gensim')

    print('Starting the API')
    app.run(host="0.0.0.0", debug=True)
