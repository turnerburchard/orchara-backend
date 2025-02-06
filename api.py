from flask import Flask, request, jsonify
from search import search_api

app = Flask(__name__)

@app.route('/api/search', methods=['POST'])
def api_search():
    payload = request.get_json()
    if not payload or 'query' not in payload or 'cluster_size' not in payload:
        return jsonify({'error': 'Parameters "query" and "cluster_size" are required.'}), 400

    query = payload['query']
    try:
        cluster_size = int(payload['cluster_size'])
    except (ValueError, TypeError):
        return jsonify({'error': '"cluster_size" must be an integer.'}), 400

    try:
        results = search_api(query, cluster_size)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
