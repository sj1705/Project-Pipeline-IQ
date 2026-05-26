"""Tests for health and root endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Root endpoint returns app name and healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "PipelineIQ" in data["message"]
    assert data["status"] == "healthy"


def test_health():
    """Health endpoint returns DB connection status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data


def test_health_db_connected():
    """Health endpoint shows database is connected."""
    response = client.get("/health")
    data = response.json()
    assert data["database"] == "connected"
