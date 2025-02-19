import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
import psycopg2
from util import get_connection

# Global configuration
INDEX_PATH = "../../orchera-etl/index/hnsw_index.bin"
MAPPING_PATH = "../../orchera-etl/index/id_mapping.json"
DIM = 384  # Dimensionality of your embeddings

# Load the SentenceTransformer model for query embeddings.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the ID mapping from disk (maps internal index to paper id).
with open(MAPPING_PATH, "r") as f:
    id_map = json.load(f)
num_elements = len(id_map)

# Initialize and load the HNSWlib index.
index = hnswlib.Index(space='cosine', dim=DIM)
# Do not call init_index here; simply load the existing index.
index.load_index(INDEX_PATH)
index.set_ef(50)  # Adjust query-time parameter as needed


def get_query_embedding(query):
    """
    Encodes the query using SentenceTransformer and normalizes the result.
    """
    embedding = model.encode(query)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def get_paper_details(paper_ids):
    """
    Given a list of paper_ids, queries PostgreSQL to retrieve paper details.
    Assumes the papers table has columns 'id', 'title', 'abstract', and 'url'.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Fetch only the necessary columns for efficiency.
    query = "SELECT id, title, abstract, url FROM public.papers WHERE id = ANY(%s)"
    cur.execute(query, (paper_ids,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    # Build a dictionary mapping paper_id to its details.
    details = {
        row[0]: {"title": row[1], "abstract": row[2], "url": row[3]}
        for row in results
    }
    return details



def search_api(query, cluster_size):
    """
    Given a query string and the desired number of neighbors (cluster_size),
    returns a list of search results. Each result is a paper with various attributes
    such as "title" and "abstract."
    """
    # Get the normalized embedding for the query.
    query_embedding = get_query_embedding(query)
    # Ensure the embedding has shape (1, DIM) as required by knn_query.
    query_embedding = np.array([query_embedding], dtype=np.float32)
    labels, distances = index.knn_query(query_embedding, k=cluster_size)

    internal_ids = labels[0]  # Array of internal index numbers.
    # Use the id_map to get the corresponding paper ids.
    paper_ids = [id_map.get(str(internal_id)) for internal_id in internal_ids]

    # Fetch additional paper details (e.g. title, abstract) from the database.
    details_dict = get_paper_details(paper_ids)

    results = []
    for i, internal_id in enumerate(internal_ids):
        paper_id = paper_ids[i]
        distance = distances[0][i]
        paper_details = details_dict.get(paper_id, {})
        result = {
            "internal_id": int(internal_id),
            "paper_id": paper_id,
            "distance": float(distance),
            "title": paper_details.get("title", ""),
            "abstract": paper_details.get("abstract", "")
        }
        results.append(result)
    return results
