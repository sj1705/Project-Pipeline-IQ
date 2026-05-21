import uuid
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import settings
from app.models.database import engine, get_db, init_db
from app.models.schemas import Base, Document, Chunk
from app.services.storage_service import storage_service
from app.pipeline.ingestion import parse_document
from app.pipeline.chunking import TextChunker
from app.pipeline.embedding import embedding_service
from app.pipeline.retrieval import vector_search
from pydantic import BaseModel
from app.pipeline.generation import llm_service
from app.routing.query_router import query_router
from app.pipeline.retrieval import vector_search, hybrid_search


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

    # Chunk the text
    chunks = chunker.chunk_text(extracted_text)

    # Generate embeddings for all chunks
    embeddings = embedding_service.generate_embeddings_batch(chunks)

    # Save document record to DB
    doc = Document(
        filename=filename,
        file_type=file_type,
        s3_path=saved_path,
        chunk_config={"chunk_size": 512, "chunk_overlap": 50},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Save chunks + embeddings to DB
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk = Chunk(
            document_id=doc.id,
            content=chunk_text,
            embedding=embedding,
            chunk_index=i,
            chunk_size=len(chunk_text),
            overlap=50,
        )
        db.add(chunk)
    db.commit()

    return {
        "document_id": str(doc.id),
        "filename": filename,
        "file_type": file_type,
        "text_length": len(extracted_text),
        "num_chunks": len(chunks),
        "message": "Document ingested and embedded successfully",
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_chunks(request: QueryRequest, db: Session = Depends(get_db)):
    results = vector_search(request.query, db, top_k=request.top_k)
    return {
        "question": request.query,
        "num_results": len(results),
        "results": results,
    }

@app.post("/query")
def query_document(request: QueryRequest, db: Session = Depends(get_db)):
    # Step 1: Retrieve relevant chunks using HYBRID search
    context_chunks = hybrid_search(request.query, db, top_k=request.top_k)

    if not context_chunks:
        return {
            "question": request.query,
            "answer": "No documents found. Please ingest relevant documents to get answers.",
            "sources": [],
        }

    # Step 2: Route query to appropriate model
    routing = query_router.classify_complexity(request.query, context_chunks)

    # Step 3: Generate answer using routed model
    llm_response = llm_service.generate_response(
        request.query, context_chunks, model_id=routing["model"]
    )

    return {
        "question": request.query,
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "complexity": routing["complexity"],
        "complexity_score": routing["score"],
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "num_sources": len(context_chunks),
        "sources": [
            {
                "content": chunk["content"][:200],
                "similarity": chunk.get("similarity", 0),
                "rrf_score": chunk.get("rrf_score", 0),
            }
            for chunk in context_chunks
        ],
    }
