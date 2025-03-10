import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from util import get_connection

# Global configuration
INDEX_PATH = "index/hnsw_index.bin" # TODO fix this
MAPPING_PATH = "index/id_mapping.json"
DIM = 384

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
    try:
        conn = get_connection()
        if not conn:
            print("Database connection failed!")
            return {}

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
    except Exception as e:
        print(f"Database error: {str(e)}")
        return {}


def search_api(query, cluster_size):
    """
    Returns up to `cluster_size` results with valid abstracts.
    If fewer results with valid abstracts exist, returns all available results.
    """
    try:
        # Start with a larger batch size and increase if needed
        multiplier = 5
        max_attempts = 3
        
        print(f"Processing search query: '{query}' with cluster_size: {cluster_size}")
        
        query_embedding = get_query_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        results = []
        attempt = 0
        
        while len(results) < cluster_size and attempt < max_attempts:
            extended_k = cluster_size * multiplier * (attempt + 1)
            max_elements = index.get_max_elements()
            k = min(extended_k, max_elements)
            
            print(f"Attempt {attempt + 1}: Querying index with k={k}")
            
            labels, distances = index.knn_query(query_embedding, k=k)
            
            # Convert internal IDs to paper IDs
            internal_ids = labels[0]
            paper_ids = [id_map.get(str(internal_id)) for internal_id in internal_ids]
            
            print(f"Found {len(paper_ids)} potential matches")
            
            # Fetch only the papers that have valid abstracts
            details_dict = get_paper_details(paper_ids)
            
            print(f"Retrieved {len(details_dict)} papers with valid abstracts")
            
            # Reset results for new attempt
            results = []
            
            # Build results in the original distance order
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
                    
                    if len(results) >= cluster_size:
                        break
            
            print(f"Processed {len(results)} valid results")
            attempt += 1
        
        if not results:
            print("Warning: No results found!")
            
        return results
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
