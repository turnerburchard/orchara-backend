import numpy as np
from sentence_transformers import SentenceTransformer
from .config import SearchConfig

class EmbeddingService:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding 