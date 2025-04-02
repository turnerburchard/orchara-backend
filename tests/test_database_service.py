import pytest
from app.services.database import DatabaseService

@pytest.mark.asyncio
async def test_database_service_initialization():
    """Test that database service initializes."""
    service = DatabaseService()
    assert isinstance(service, DatabaseService)

@pytest.mark.asyncio
async def test_get_papers_empty_list():
    """Test that empty paper list returns empty result."""
    service = DatabaseService()
    papers = await service.get_papers([])
    assert isinstance(papers, list)
    assert len(papers) == 0 