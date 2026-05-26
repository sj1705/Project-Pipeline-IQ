"""Tests for the optimization agent and its tools."""
import pytest
import json
from app.agents.optimizer import read_query_metrics, read_past_configs, propose_config


def test_read_query_metrics_no_data():
    """Test read_query_metrics returns no_data when DB is empty of logs."""
    result = read_query_metrics.invoke({"window": 5})
    data = json.loads(result)
    # Either returns metrics or no_data status
    assert "total_queries_analyzed" in data or data.get("status") == "no_data"


def test_read_past_configs_no_data():
    """Test read_past_configs returns message when no configs exist."""
    result = read_past_configs.invoke({"limit": 5})
    data = json.loads(result)
    # Either returns configs list or no_configs status
    assert isinstance(data, list) or data.get("status") == "no_configs"


def test_propose_config_clamps_values():
    """Test propose_config clamps values to valid bounds."""
    # Try extreme values — should be clamped
    result = propose_config.invoke({
        "top_k": 100,  # max is 10
        "rerank_weight": 2.0,  # max is 0.9
        "routing_threshold": -1.0,  # min is 0.3
        "retry_threshold": 0.1,  # min is 0.5
        "reasoning": "Testing bounds clamping",
    })
    data = json.loads(result)

    assert data["status"] == "config_proposed"
    assert data["config"]["top_k"] == 10  # clamped to max
    assert data["config"]["rerank_weight"] == 0.9  # clamped to max
    assert data["config"]["routing_threshold"] == 0.3  # clamped to min
    assert data["config"]["retry_threshold"] == 0.5  # clamped to min


def test_propose_config_valid_values():
    """Test propose_config accepts valid values."""
    result = propose_config.invoke({
        "top_k": 7,
        "rerank_weight": 0.6,
        "routing_threshold": 0.5,
        "retry_threshold": 0.7,
        "reasoning": "Testing valid config proposal",
    })
    data = json.loads(result)

    assert data["status"] == "config_proposed"
    assert data["config"]["top_k"] == 7
    assert data["config"]["rerank_weight"] == 0.6
    assert data["config"]["routing_threshold"] == 0.5
    assert data["config"]["retry_threshold"] == 0.7
    assert "version" in data


def test_optimize_endpoint():
    """Test /optimize endpoint triggers the agent."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/optimize")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
