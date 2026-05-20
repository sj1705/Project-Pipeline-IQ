import uuid
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import settings
from app.models.database import engine, get_db, init_db
from app.models.schemas import Base, Document
from app.services.storage_service import storage_service
from app.pipeline.ingestion import parse_document
from app.pipeline.chunking import TextChunker


app = FastAPI(
    title=settings.app_name,
    description="Self-Optimizing RAG Orchestration System",
    version="0.1.0",
)

init_db()

chunker = TextChunker(chunk_size=512, chunk_overlap=50)



@app.get("/")
def root():
    return {"message": f"{settings.app_name} is running", "status": "healthy"}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    return {
        "status": "ok",
        "debug": settings.debug,
        "database": db_status,
    }


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate file type
    filename = file.filename
    file_type = filename.rsplit(".", 1)[-1].lower()

    if file_type not in ["pdf", "docx", "html"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")

    # Save file locally
    saved_path = storage_service.save_file(file, filename)

    # Extract text
    extracted_text = parse_document(saved_path, file_type)

    # Save document record to DB
    doc = Document(
        filename=filename,
        file_type=file_type,
        s3_path=saved_path,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "document_id": str(doc.id),
        "filename": filename,
        "file_type": file_type,
        "text_length": len(extracted_text),
        "preview": extracted_text[:500],  # first 500 chars as preview
    }