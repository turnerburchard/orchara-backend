import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from util import get_connection
from collections import defaultdict
import re
import os

# Global configuration
INDEX_PATH = os.environ.get('INDEX_PATH', 'index/hnsw_index.bin')
MAPPING_PATH = os.environ.get('MAPPING_PATH', 'index/id_mapping.json')
DIM = 384

# Search hyperparameters
SEARCH_MULTIPLIER = 5  # When searching, fetch 5x more results than needed initially.
                      # This helps ensure we have enough results after filtering out papers
                      # with missing/invalid abstracts. Increases with each attempt.
MAX_SEARCH_ATTEMPTS = 3  # Maximum number of attempts to get enough results
HNSW_EF = 50  # HNSW search quality parameter (higher = more accurate but slower)
REQUIRE_ABSTRACT = True  # Whether to only return papers with valid abstracts

# Diversity scoring hyperparameters
MIN_KEYWORD_LENGTH = 3  # Minimum length for a word to be considered a keyword
STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 
    'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
}

# Result ranking hyperparameters
SEMANTIC_WEIGHT = 0.4  # Weight for semantic similarity in final score
KEYWORD_WEIGHT = 0.5   # Weight for keyword relevance to search query
DIVERSITY_WEIGHT = 0.1  # Weight for diversity in final score
RESULTS_MULTIPLIER = 10  # Fetch 2x more results than requested to have a larger pool
                       # for scoring. This ensures we can select diverse papers
                       # even if some highly similar papers are in the top results.

# Load the SentenceTransformer model for query embeddings.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the ID mapping from disk.
with open(MAPPING_PATH, "r") as f:
    id_map = json.load(f)

# Initialize and load the HNSWlib index.
index = hnswlib.Index(space='cosine', dim=DIM)
index.load_index(INDEX_PATH)
index.set_ef(HNSW_EF)  # Set HNSW search quality parameter


def get_query_embedding(query):
    embedding = model.encode(query)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def extract_keywords(text):
    """Extract meaningful keywords from text, excluding common words"""
    if not text:
        return []
    # Convert to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in STOP_WORDS and len(word) > MIN_KEYWORD_LENGTH]
    
    return keywords


def calculate_keyword_relevance(query, text):
    """Calculate how relevant the text is to the search query based on keyword overlap"""
    if not text:
        return 0.0
        
    query_keywords = set(extract_keywords(query))
    text_keywords = set(extract_keywords(text))
    
    if not query_keywords or not text_keywords:
        return 0.0
    
    # Calculate Jaccard similarity between query and text keywords
    overlap = len(query_keywords.intersection(text_keywords))
    total = len(query_keywords.union(text_keywords))
    
    return overlap / total if total > 0 else 0.0


def calculate_diversity_score(results):
    """Calculate diversity score based on keyword overlap"""
    if not results:
        return []
    
    # Extract keywords from all abstracts
    all_keywords = []
    for result in results:
        keywords = extract_keywords(result.get('abstract', ''))
        all_keywords.extend(keywords)
    
    # Count keyword frequencies
    keyword_counts = defaultdict(int)
    for keyword in all_keywords:
        keyword_counts[keyword] += 1
    
    # Calculate diversity scores (lower frequency = higher diversity)
    diversity_scores = []
    for result in results:
        keywords = extract_keywords(result.get('abstract', ''))
        # Average frequency of keywords in this result
        avg_frequency = sum(keyword_counts[k] for k in keywords) / len(keywords) if keywords else 0
        # Convert to diversity score (lower frequency = higher diversity)
        diversity_score = 1 / (1 + avg_frequency)
        diversity_scores.append(diversity_score)
    
    return diversity_scores


def get_paper_details(paper_ids):
    """
    Fetches paper details, optionally filtering for valid abstracts.
    """
    try:
        conn = get_connection()
        if not conn:
            print("Database connection failed!")
            return {}

        cur = conn.cursor()
        if REQUIRE_ABSTRACT:
            query = """
                SELECT id, title, abstract, url
                FROM public.papers
                WHERE id = ANY(%s)
                  AND abstract IS NOT NULL
                  AND abstract <> ''
            """
        else:
            query = """
                SELECT id, title, abstract, url
                FROM public.papers
                WHERE id = ANY(%s)
            """
            
        cur.execute(query, (paper_ids,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {
            row[0]: {
                "title": row[1], 
                "abstract": row[2] or "",  # Convert None to empty string
                "url": row[3]
            }
            for row in rows
        }
    except Exception as e:
        print(f"Database error: {str(e)}")
        return {}


def search_api(query, cluster_size):
    """
    Returns up to `cluster_size` results, using semantic similarity,
    keyword matching, and diversity scoring.
    """
    try:
        print(f"Processing search query: '{query}' with cluster_size: {cluster_size}")
        
        query_embedding = get_query_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        results = []
        attempt = 0
        
        while len(results) < cluster_size * RESULTS_MULTIPLIER and attempt < MAX_SEARCH_ATTEMPTS:
            extended_k = cluster_size * SEARCH_MULTIPLIER * (attempt + 1)
            max_elements = index.get_max_elements()
            k = min(extended_k, max_elements)
            
            print(f"Attempt {attempt + 1}: Querying index with k={k}")
            
            labels, distances = index.knn_query(query_embedding, k=k)
            
            # Convert internal IDs to paper IDs
            internal_ids = labels[0]
            paper_ids = [id_map.get(str(internal_id)) for internal_id in internal_ids]
            
            print(f"Found {len(paper_ids)} potential matches")
            
            # Fetch papers with optional abstract filtering
            details_dict = get_paper_details(paper_ids)
            
            print(f"Retrieved {len(details_dict)} papers")
            
            # Reset results for new attempt
            results = []
            
            # Build results with semantic similarity scores
            for i, internal_id in enumerate(internal_ids):
                paper_id = paper_ids[i]
                distance = distances[0][i]
                if paper_id in details_dict:
                    paper_details = details_dict[paper_id]
                    # Calculate keyword relevance score
                    keyword_score = calculate_keyword_relevance(query, paper_details["abstract"])
                    
                    results.append({
                        "internal_id": int(internal_id),
                        "paper_id": paper_id,
                        "semantic_score": 1 - float(distance),  # Convert distance to similarity score
                        "keyword_score": keyword_score,
                        "title": paper_details["title"],
                        "abstract": paper_details["abstract"],
                        "url": paper_details["url"],
                    })
            
            print(f"Processed {len(results)} valid results")
            attempt += 1
        
        if not results:
            print("Warning: No results found!")
            return []
        
        # Calculate diversity scores
        diversity_scores = calculate_diversity_score(results)
        
        # Combine all scores
        for i, result in enumerate(results):
            result['diversity_score'] = diversity_scores[i]
            # Final score: weighted combination of semantic, keyword, and diversity scores
            result['final_score'] = (
                SEMANTIC_WEIGHT * result['semantic_score'] +
                KEYWORD_WEIGHT * result['keyword_score'] +
                DIVERSITY_WEIGHT * diversity_scores[i]
            )
        
        # Sort by final score and take top results
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:cluster_size]
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
