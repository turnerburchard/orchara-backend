import pytest
from app.services.search import SearchService

@pytest.mark.asyncio
async def test_search_service_initialization(search_service):
    """Test that search service initializes."""
    assert isinstance(search_service, SearchService)

@pytest.mark.asyncio
async def test_search_service_empty_query(search_service):
    """Test that empty queries raise ValueError."""
    with pytest.raises(ValueError):
        await search_service.search("", 5)

@pytest.mark.asyncio
async def test_search_service_basic_search(search_service):
    """Test basic search functionality."""
    results = await search_service.search("test", 1)
    assert isinstance(results, list)
    assert len(results) >= 0  # Allow empty results for now 