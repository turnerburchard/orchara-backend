from flask import Flask, request, jsonify
from search import search_api
from summarize import Summarizer
from flask_cors import CORS
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from util import get_connection

app = Flask(__name__)

# Configure CORS for development and production
CORS(app, 
     resources={r"/*": {
         "origins": [
             "http://localhost:5173",  # Vite dev server
             "https://orchara.com",
             "https://www.orchara.com",
             "https://api.orchara.com"
         ],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True
     }})

@app.route('/health', methods=['GET'])
def health_check():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM papers")
            row_count = cur.fetchone()[0]
        return jsonify({"rows_loaded": row_count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

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
    app.run(host='0.0.0.0', port=5001, debug=True)

