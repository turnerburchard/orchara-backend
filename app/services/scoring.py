from typing import List, Dict, Any
import numpy as np
from app.models import SearchResult

class ScoringService:
    def __init__(self):
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4

    def calculate_semantic_score(self, query_embedding: np.ndarray, result_embedding: np.ndarray) -> float:
        """Calculate semantic similarity score between query and result."""
        return float(np.dot(query_embedding, result_embedding) / 
                    (np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)))

    def calculate_keyword_score(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate keyword matching score."""
        query_words = set(query.lower().split())
        title_words = set(result.get('title', '').lower().split())
        abstract_words = set(result.get('abstract', '').lower().split())
        
        title_matches = len(query_words & title_words) / len(query_words)
        abstract_matches = len(query_words & abstract_words) / len(query_words)
        
        return 0.7 * title_matches + 0.3 * abstract_matches

    def calculate_final_scores(self, query: str, query_embedding: np.ndarray, 
                             results: List[Dict[str, Any]], result_embeddings: np.ndarray) -> List[float]:
        """Calculate final scores for search results."""
        scores = []
        for i, result in enumerate(results):
            semantic_score = self.calculate_semantic_score(query_embedding, result_embeddings[i])
            keyword_score = self.calculate_keyword_score(query, result)
            
            final_score = (self.semantic_weight * semantic_score + 
                         self.keyword_weight * keyword_score)
            scores.append(final_score)
        return scores 