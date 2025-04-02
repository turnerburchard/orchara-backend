import pytest
from app.services.embedding import EmbeddingService

@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test that embedding service initializes."""
    service = EmbeddingService()
    assert isinstance(service, EmbeddingService)
    assert service.model is not None

@pytest.mark.asyncio
async def test_get_query_embedding():
    service = EmbeddingService()
    query = "machine learning"
    embedding = service.get_embedding(query)
    assert embedding is not None
    assert embedding.shape[0] == service.get_embedding_dimension()

@pytest.mark.asyncio
async def test_get_query_embedding_empty_query():
    service = EmbeddingService()
    embedding = service.get_embedding("")
    assert embedding is not None
    assert embedding.shape[0] == service.get_embedding_dimension() 