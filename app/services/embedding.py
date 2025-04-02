from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str, normalize: bool = False) -> np.ndarray:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        if normalize:
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        return embedding

    def get_embeddings(self, texts: List[str], normalize: bool = False) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings

    async def get_embedding_async(self, text: str, normalize: bool = False) -> np.ndarray:
        """Asynchronously generate embedding for a single text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text, normalize)

    async def get_embeddings_async(self, texts: List[str], normalize: bool = False) -> np.ndarray:
        """Asynchronously generate embeddings for multiple texts."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embeddings, texts, normalize)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension() 