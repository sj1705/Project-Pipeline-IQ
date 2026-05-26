"""
Ingest ALL CUAD contract PDFs for testing.
Run: python scripts/ingest_cuad.py
"""

import os
import requests
import glob
import time

API_URL = "http://127.0.0.1:8003"
CUAD_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset")


def check_api():
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.status_code == 200:
            print("✅ API is running")
            return True
    except:
        pass
    print("❌ API not running. Start: python -m uvicorn app.main:app --port 8003")
    return False


def ingest_file(filepath):
    filename = os.path.basename(filepath)
    display_name = filename[:60] + "..." if len(filename) > 60 else filename
    print(f"   📄 {display_name}", end=" ")

    content_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "html": "text/html",
        "txt": "text/plain",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    ext = filename.rsplit(".", 1)[-1].lower()
    content_type = content_types.get(ext, "application/octet-stream")

    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{API_URL}/ingest",
            files={"file": (filename, f, content_type)},
        )

    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ {data['num_chunks']} chunks")
        return True
    else:
        print(f"❌ {resp.status_code}")
        return False


def main():
    print("📚 CUAD Dataset — Full Ingestion")
    print("=" * 50)

    if not check_api():
        return

    if not os.path.exists(CUAD_DIR):
        print(f"❌ Dataset not found at: {CUAD_DIR}")
        return

    # Find ALL supported file types recursively
    supported_extensions = ["*.pdf", "*.docx", "*.html", "*.txt", "*.xlsx"]
    all_files = []
    for ext in supported_extensions:
        all_files.extend(glob.glob(os.path.join(CUAD_DIR, "**", ext), recursive=True))

    print(f"\n📁 Found {len(all_files)} files in {CUAD_DIR}")
    # Show breakdown by type
    by_type = {}
    for f in all_files:
        ext = f.rsplit(".", 1)[-1].lower()
        by_type[ext] = by_type.get(ext, 0) + 1
    for ext, count in by_type.items():
        print(f"   .{ext}: {count}")
    print(f"\n   Starting ingestion...\n")

    start_time = time.time()
    success = 0
    failed = 0

    for i, file_path in enumerate(all_files, 1):
        category = os.path.basename(os.path.dirname(file_path))
        print(f"[{i}/{len(all_files)}] ({category})")
        if ingest_file(file_path):
            success += 1
        else:
            failed += 1

    elapsed = round(time.time() - start_time, 1)
    print(f"\n{'=' * 50}")
    print(f"✅ Done in {elapsed}s")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"   Total files: {len(all_files)}")


if __name__ == "__main__":
    main()
