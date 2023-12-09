from flask import Flask, request, jsonify
from flask_cors import CORS
from chroma import compare_embedding_with_url, store_image_embedding_from_url

app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    current_url = data.get('current')
    before_urls = data.get('before')

    if not current_url or not before_urls:
        return 'Missing data', 400

    # Store and compare image embedding
    store_image_embedding_from_url(current_url)
    similarities = compare_embedding_with_url(current_url, before_urls)

    return jsonify(similarities)


if __name__ == '__main__':
    app.run(debug=True, port=3001)
