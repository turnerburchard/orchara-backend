from flask import Flask, request, jsonify
from search import search_api
from summarize import Summarizer
from flask_cors import CORS
import os

app = Flask(__name__)

# Simplify CORS configuration - for development purposes
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5001", "http://127.0.0.1:5001"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/api/search', methods=['POST'])
def api_search():
    print("\n=== New Search Request ===")
    print("Request received:", request)
    print("Request headers:", dict(request.headers))
    print("Request data:", request.get_data())
    
    try:
        payload = request.get_json()
        print("Parsed JSON payload:", payload)
    except Exception as e:
        print("Error parsing JSON:", str(e))
        return jsonify({'error': 'Invalid JSON payload'}), 400

    if not payload or 'query' not in payload or 'cluster_size' not in payload:
        print("Missing required parameters")
        return jsonify({'error': 'Parameters "query" and "cluster_size" are required.'}), 400

    query = payload['query']
    try:
        cluster_size = int(payload['cluster_size'])
    except (ValueError, TypeError) as e:
        print("Invalid cluster_size:", str(e))
        return jsonify({'error': '"cluster_size" must be an integer.'}), 400

    try:
        print(f"Calling search_api with query='{query}', cluster_size={cluster_size}")
        results = search_api(query, cluster_size)
        print("Search results:", results)
        return jsonify({'results': results})
    except Exception as e:
        print("Search error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    payload = request.get_json()
    if not payload or 'text' not in payload:
        return jsonify({'error': 'Parameter "text" is required.'}), 400

    text = payload['text']
    summarizer = Summarizer()
    try:
        response = summarizer.summarize(text)
        return jsonify({'summary': response})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

