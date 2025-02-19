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

# Load the ID mapping from disk.
with open(MAPPING_PATH, "r") as f:
    id_map = json.load(f)

# Initialize and load the HNSWlib index.
index = hnswlib.Index(space='cosine', dim=DIM)
index.load_index(INDEX_PATH)
index.set_ef(50)  # Adjust query-time parameter as needed


def get_query_embedding(query):
    embedding = model.encode(query)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def get_paper_details(paper_ids):
    """
    Fetches only rows with a nonempty abstract.
    """
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT id, title, abstract, url
        FROM public.papers
        WHERE id = ANY(%s)
          AND abstract IS NOT NULL
          AND abstract <> ''
    """
    cur.execute(query, (paper_ids,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {
        row[0]: {"title": row[1], "abstract": row[2], "url": row[3]}
        for row in rows
    }


def search_api(query, cluster_size):
    """
    Returns exactly `cluster_size` results with valid abstracts.
    """
    # Get a larger batch than cluster_size from the index (e.g., 5x).
    extended_k = 5 * cluster_size

    query_embedding = get_query_embedding(query)
    query_embedding = np.array([query_embedding], dtype=np.float32)
    labels, distances = index.knn_query(query_embedding, k=extended_k)

    # Convert internal IDs to paper IDs
    internal_ids = labels[0]
    paper_ids = [id_map.get(str(internal_id)) for internal_id in internal_ids]

    # Fetch only the papers that have valid abstracts
    details_dict = get_paper_details(paper_ids)

    # Build results in the original distance order
    results = []
    for i, internal_id in enumerate(internal_ids):
        paper_id = paper_ids[i]
        distance = distances[0][i]
        if paper_id in details_dict:
            paper_details = details_dict[paper_id]
            results.append({
                "internal_id": int(internal_id),
                "paper_id": paper_id,
                "distance": float(distance),
                "title": paper_details["title"],
                "abstract": paper_details["abstract"],
                "url": paper_details["url"],
            })

    # Keep only the top `cluster_size` results
    return results[:cluster_size]
