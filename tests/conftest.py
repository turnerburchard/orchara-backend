import pytest
from fastapi.testclient import TestClient
from app.services.search import SearchService
from app.services.search.config import SearchConfig

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up minimal test environment variables."""
    yield

@pytest.fixture
def test_client():
    """Test client for API endpoints."""
    from app.api.routes import app
    return TestClient(app)

@pytest.fixture
def search_service():
    """Minimal search service for testing."""
    config = SearchConfig(
        DIM=384,
        SEARCH_MULTIPLIER=1,
        MAX_SEARCH_ATTEMPTS=1,
        HNSW_EF=10,
        REQUIRE_ABSTRACT=False,
        MIN_KEYWORD_LENGTH=3,
        SEMANTIC_WEIGHT=0.4,
        KEYWORD_WEIGHT=0.5,
        DIVERSITY_WEIGHT=0.1,
        RESULTS_MULTIPLIER=1
    )
    return SearchService(config) 