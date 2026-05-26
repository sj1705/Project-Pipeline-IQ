"""Tests for document ingestion endpoint."""
import pytest
import os
import tempfile
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_test_pdf():
    """Create a minimal test PDF file."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is a test document about machine learning. "
                     "Machine learning is a subset of artificial intelligence. "
                     "It allows systems to learn from data without being explicitly programmed.")

    # Use a named path instead of NamedTemporaryFile (Windows permission issue)
    pdf_path = os.path.join(tempfile.gettempdir(), "pipelineiq_test.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_ingest_pdf():
    """Test PDF upload and ingestion."""
    pdf_path = create_test_pdf()

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("test_doc.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_doc.pdf"
    assert data["file_type"] == "pdf"
    assert data["num_chunks"] > 0
    assert data["text_length"] > 0
    assert "document_id" in data

    # Cleanup
    os.unlink(pdf_path)


def test_ingest_unsupported_type():
    """Test rejection of unsupported file types."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    tmp.write(b"Hello world")
    tmp.close()

    with open(tmp.name, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("test.txt", f, "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

    os.unlink(tmp.name)


def test_ingest_html():
    """Test HTML file ingestion."""
    html_content = b"""
    <html>
    <body>
    <h1>Test Document</h1>
    <p>This is a test HTML document about artificial intelligence and natural language processing.
    NLP is used in many applications including chatbots, translation, and search engines.</p>
    </body>
    </html>
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    tmp.write(html_content)
    tmp.close()

    with open(tmp.name, "rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("test.html", f, "text/html")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["file_type"] == "html"
    assert data["num_chunks"] > 0

    os.unlink(tmp.name)
