"""
Seed script — loads sample documents into PipelineIQ.
Run: python scripts/seed.py
"""

import requests
import os
import sys

API_URL = "http://127.0.0.1:8003"

# Sample documents to ingest
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


def check_api():
    """Verify the API is running."""
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.status_code == 200:
            print("✅ API is running")
            return True
    except:
        pass
    print("❌ API is not running. Start it with: python -m uvicorn app.main:app --port 8003")
    return False


def ingest_file(filepath):
    """Upload a file to the /ingest endpoint."""
    filename = os.path.basename(filepath)
    print(f"   Ingesting: {filename}...", end=" ")

    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{API_URL}/ingest",
            files={"file": (filename, f)},
        )

    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ {data['num_chunks']} chunks created")
        return True
    else:
        print(f"❌ Error: {resp.text}")
        return False


def run_sample_queries():
    """Run a few sample queries to populate query_logs."""
    queries = [
        "What is the total budget allocation?",
        "What are the key economic indicators?",
        "How much is allocated for infrastructure?",
        "What is the fiscal deficit target?",
        "What are the major tax reforms?",
    ]

    print("\n📝 Running sample queries to populate metrics...")
    for q in queries:
        print(f"   Query: {q[:50]}...", end=" ")
        resp = requests.post(f"{API_URL}/query-optimized", json={"query": q})
        if resp.status_code == 200:
            print("✅")
        else:
            print(f"❌ {resp.status_code}")


def main():
    print("🌱 PipelineIQ Seed Script")
    print("=" * 40)

    if not check_api():
        sys.exit(1)

    # Find all PDFs/DOCX/HTML in uploads folder
    print("\n📄 Looking for documents in uploads/...")
    files_found = []
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.endswith((".pdf", ".docx", ".html")):
                files_found.append(os.path.join(UPLOAD_DIR, f))

    if not files_found:
        print("   No documents found in uploads/")
        print("   Add PDF/DOCX/HTML files to the uploads/ folder and run again.")
        sys.exit(0)

    print(f"   Found {len(files_found)} document(s)")

    # Ingest each file
    print("\n📥 Ingesting documents...")
    for filepath in files_found:
        ingest_file(filepath)

    # Run sample queries
    run_sample_queries()

    # Trigger optimizer
    print("\n🧠 Running optimizer...")
    resp = requests.get(f"{API_URL}/optimize")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   Status: {data.get('status', 'unknown')}")

    print("\n✅ Seed complete! Open dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()