"""Tests for A/B test service."""
import pytest
from unittest.mock import MagicMock, patch
from app.services.ab_test_service import ABTestService


def test_defaults_when_no_config():
    """Test that defaults are returned when no config exists."""
    service = ABTestService()

    # Mock DB session with no configs
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    result = service.get_test_state(mock_db)

    assert result["active"] == False
    assert result["use_config"]["top_k"] == 5
    assert result["use_config"]["rerank_weight"] == 0.5
    assert result["use_config"]["routing_threshold"] == 0.5
    assert result["use_config"]["retry_threshold"] == 0.7
    assert result["config_version"] == 0


def test_no_ab_test_when_no_proposed():
    """Test that A/B test is not active when there's no proposed config."""
    service = ABTestService()

    # Mock: active config exists, no proposed
    mock_db = MagicMock()
    mock_active = MagicMock()
    mock_active.version = 1
    mock_active.top_k = 5
    mock_active.rerank_weight = 0.5
    mock_active.routing_threshold = 0.5
    mock_active.retry_threshold = 0.7
    mock_active.is_active = True

    # First query().filter() returns active, second returns None (no proposed)
    mock_db.query.return_value.filter.return_value.first.return_value = mock_active
    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    result = service.get_test_state(mock_db)

    assert result["active"] == False
    assert result["config_version"] == 1


def test_defaults_dict():
    """Test that _defaults returns correct structure."""
    service = ABTestService()
    defaults = service._defaults()

    assert defaults["top_k"] == 5
    assert defaults["rerank_weight"] == 0.5
    assert defaults["routing_threshold"] == 0.5
    assert defaults["retry_threshold"] == 0.7


def test_config_to_dict():
    """Test that _config_to_dict extracts correct fields."""
    service = ABTestService()

    mock_config = MagicMock()
    mock_config.top_k = 7
    mock_config.rerank_weight = 0.8
    mock_config.routing_threshold = 0.6
    mock_config.retry_threshold = 0.7

    result = service._config_to_dict(mock_config)

    assert result == {
        "top_k": 7,
        "rerank_weight": 0.8,
        "routing_threshold": 0.6,
        "retry_threshold": 0.7,
    }


def test_queries_per_config_constant():
    """Test that QUERIES_PER_CONFIG is set to 10."""
    service = ABTestService()
    assert service.QUERIES_PER_CONFIG == 10


def test_ab_test_endpoint():
    """Test /ab-test endpoint returns status."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/ab-test")
    assert response.status_code == 200
    data = response.json()
    assert "active" in data
    assert "use_config" in data
    assert "config_version" in data
