from fastapi.testclient import TestClient

def test_search_empty_query(test_client: TestClient):
    """Test that empty search query returns 422."""
    response = test_client.post(
        "/api/search",
        json={"query": "", "cluster_size": 5}
    )
    assert response.status_code == 422 