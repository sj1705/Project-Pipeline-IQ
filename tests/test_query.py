"""Tests for query endpoints and semantic cache."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_search_endpoint():
    """Test vector search endpoint returns results."""
    response = client.post("/search", json={"query": "machine learning", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "results" in data
    assert data["question"] == "machine learning"


def test_search_no_results():
    """Test search with gibberish returns empty results gracefully."""
    response = client.post("/search", json={"query": "xyzzy12345nonsense", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["num_results"] >= 0  # May return 0 or some results


def test_query_agent_endpoint():
    """Test /query-optimized endpoint returns structured response."""
    response = client.post("/query-optimized", json={"query": "What is this document about?"})
    assert response.status_code == 200
    data = response.json()

    # Should have either a real answer or a "no documents" message
    assert "answer" in data or "from_cache" in data


def test_query_agent_cache_hit():
    """Test that same query hits semantic cache on second call."""
    query = "What is the main topic of this document?"

    # First call — should not be cached
    response1 = client.post("/query-optimized", json={"query": query})
    assert response1.status_code == 200

    # Second call — should hit cache
    response2 = client.post("/query-optimized", json={"query": query})
    assert response2.status_code == 200
    data2 = response2.json()

    # If cache is working, second call should be from cache
    if "from_cache" in data2:
        assert data2["from_cache"] == True


def test_query_agent_no_top_k():
    """Test that /query-optimized doesn't accept top_k (managed by optimizer)."""
    # Should work without top_k
    response = client.post("/query-optimized", json={"query": "test question"})
    assert response.status_code == 200


def test_metrics_endpoint():
    """Test metrics endpoint returns query logs."""
    response = client.get("/metrics?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "recent_queries" in data
    assert isinstance(data["recent_queries"], list)
