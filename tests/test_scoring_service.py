import pytest
from app.services.search.scoring import ScoringService
from app.services.search.config import SearchConfig

@pytest.mark.asyncio
async def test_scoring_service_initialization():
    """Test that scoring service initializes."""
    config = SearchConfig()
    service = ScoringService(config)
    assert isinstance(service, ScoringService)

@pytest.mark.asyncio
async def test_keyword_relevance_empty():
    """Test keyword relevance with empty inputs."""
    config = SearchConfig()
    service = ScoringService(config)
    score = await service.calculate_keyword_relevance("", "")
    assert isinstance(score, float)
    assert 0 <= score <= 1

@pytest.mark.asyncio
async def test_calculate_keyword_score():
    """Test keyword score calculation."""
    config = SearchConfig()
    service = ScoringService(config)
    score = await service.calculate_keyword_relevance("test", "test document")
    assert isinstance(score, float)
    assert 0 <= score <= 1 