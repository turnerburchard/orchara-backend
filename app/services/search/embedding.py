import numpy as np
from sentence_transformers import SentenceTransformer
from .config import SearchConfig
import asyncio

class EmbeddingService:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def get_query_embedding(self, query: str) -> np.ndarray:
        # Run the model inference in a thread pool since it's CPU-bound
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.model.encode, query)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding 