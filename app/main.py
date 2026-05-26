from dotenv import load_dotenv
load_dotenv()

import uuid
import asyncio
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.config import settings
from app.models.database import engine, get_db, init_db
from app.models.schemas import Base, Document, Chunk, QueryLog, PipelineConfig
from app.services.storage_service import storage_service
from app.pipeline.ingestion import parse_document
from app.pipeline.chunking import TextChunker
from app.pipeline.embedding import embedding_service
from app.pipeline.retrieval import vector_search, hybrid_search
from app.pipeline.generation import llm_service
from app.routing.query_router import query_router
from app.evaluation.ragas_eval import rag_evaluator
from app.evaluation.latency_tracker import LatencyTracker
from app.evaluation.cost_tracker import cost_tracker
from app.agents.optimizer import optimization_agent
from app.services.ab_test_service import ab_test_service
from pydantic import BaseModel
from app.models.schemas import Base, Document, Chunk, QueryLog, PipelineConfig, QueryCache
from sqlalchemy import text as sql_text


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
async def health_check(db: Session = Depends(get_db)):
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
async def ingest_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Validate file type
    filename = file.filename
    file_type = filename.rsplit(".", 1)[-1].lower()

    if file_type not in ["pdf", "docx", "html", "txt", "xlsx"]:
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

        # Invalidate query cache — new document means old answers may be stale
    db.execute(sql_text("TRUNCATE TABLE query_cache"))
    db.commit()

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


@app.post("/ingest/batch")
async def ingest_multiple(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload and ingest multiple documents at once."""
    results = []

    for file in files:
        filename = file.filename
        file_type = filename.rsplit(".", 1)[-1].lower()

        if file_type not in ["pdf", "docx", "html", "txt", "xlsx"]:
            results.append({"filename": filename, "status": "skipped", "reason": f"Unsupported type: {file_type}"})
            continue

        try:
            saved_path = storage_service.save_file(file, filename)
            extracted_text = parse_document(saved_path, file_type)
            chunks = chunker.chunk_text(extracted_text)
            embeddings = embedding_service.generate_embeddings_batch(chunks)

            doc = Document(
                filename=filename,
                file_type=file_type,
                s3_path=saved_path,
                chunk_config={"chunk_size": 512, "chunk_overlap": 50},
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

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

            results.append({
                "filename": filename,
                "status": "success",
                "document_id": str(doc.id),
                "num_chunks": len(chunks),
            })
        except Exception as e:
            results.append({"filename": filename, "status": "failed", "reason": str(e)})

    # Invalidate cache — new documents ingested
    db.execute(text("TRUNCATE TABLE query_cache"))
    db.commit()

    return {
        "total_files": len(files),
        "results": results,
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class AgentQueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_chunks(request: QueryRequest, db: Session = Depends(get_db)):
    results = vector_search(request.query, db, top_k=request.top_k)
    return {
        "question": request.query,
        "num_results": len(results),
        "results": results,
    }

@app.post("/query")
async def query_document(request: QueryRequest, db: Session = Depends(get_db)):
    # Initialize latency tracker for this request
    tracker = LatencyTracker()

    # Step 1: Retrieve relevant chunks using hybrid search
    tracker.start("retrieval")
    context_chunks = hybrid_search(request.query, db, top_k=request.top_k)
    tracker.end("retrieval")

    if not context_chunks:
        return {
            "question": request.query,
            "answer": "No documents found. Please ingest relevant documents to get answers.",
            "sources": [],
        }

    # Step 2: Route query to appropriate model
    tracker.start("routing")
    routing = query_router.classify_complexity(request.query, context_chunks)
    tracker.end("routing")

    # Step 3: Generate answer using routed model
    tracker.start("generation")
    llm_response = llm_service.generate_response(
        request.query, context_chunks, model_id=routing["model"]
    )
    tracker.end("generation")

    # Step 4: Evaluate response quality
    tracker.start("evaluation")
    contexts = [chunk["content"] for chunk in context_chunks]
    from ragas import SingleTurnSample
    sample = SingleTurnSample(
        user_input=request.query,
        response=llm_response["answer"],
        retrieved_contexts=contexts,
    )
    eval_scores = await rag_evaluator._async_evaluate(sample)
    tracker.end("evaluation")

        # Step 5: Calculate cost
    query_cost = cost_tracker.calculate_cost(
        model_id=llm_response["model_used"],
        input_tokens=llm_response["input_tokens"],
        output_tokens=llm_response["output_tokens"],


    )


    # Step 6: Store metrics in query_logs
    query_log = QueryLog(
        query=request.query,
        response=llm_response["answer"],
        model_used=llm_response["model_used"],
        latency_ms=tracker.get_total(),
        token_count=llm_response["input_tokens"] + llm_response["output_tokens"],
        cost=query_cost["total_cost_usd"],
        retrieval_scores={
            "num_sources": len(context_chunks),
            "top_similarity": context_chunks[0].get("similarity", 0) if context_chunks else 0,
            "top_rerank_score": context_chunks[0].get("rerank_score", 0) if context_chunks else 0,
        },
        evaluation_scores=eval_scores,
    )
    db.add(query_log)
    db.commit()

    return {
        "question": request.query,
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "complexity": routing["complexity"],
        "complexity_score": routing["score"],
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "evaluation": eval_scores,
        "latency": tracker.get_report(),
        "cost": query_cost,
        "num_sources": len(context_chunks),
        "sources": [
            {
                "content": chunk["content"][:200],
                "similarity": chunk.get("similarity", 0),
                "rrf_score": chunk.get("rrf_score", 0),
                "rerank_score": chunk.get("rerank_score", 0),
            }
            for chunk in context_chunks
        ],
    }


@app.get("/metrics")
async def get_metrics(limit: int = 10, db: Session = Depends(get_db)):
    """Get recent query logs with all metrics."""
    from app.models.schemas import QueryLog
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit).all()

    return {
        "total_queries": db.query(QueryLog).count(),
        "recent_queries": [
            {
                "id": str(log.id),
                "query": log.query,
                "model_used": log.model_used,
                "latency_ms": log.latency_ms,
                "token_count": log.token_count,
                "cost": log.cost,
                "evaluation_scores": log.evaluation_scores,
                "created_at": str(log.created_at),
            }
            for log in logs
        ],
    }

def get_active_config(db: Session) -> dict:
    """Read config via A/B test service — returns which config to use and version."""
    return ab_test_service.get_test_state(db)


async def run_optimizer_background():
    """Run the optimizer agent in background (non-blocking)."""
    from app.models.database import SessionLocal
    try:
        db = SessionLocal()
        await optimization_agent.run(db)
        db.close()
    except Exception as e:
        print(f"[Optimizer] Background run failed: {e}")


async def run_eval_and_log_background(
    query: str,
    answer: str,
    contexts: list,
    model_used: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    context_chunks: list,
    config_version: int,
):
    """Run RAGAS evaluation and log to DB in background. User doesn't wait for this."""
    from app.models.database import SessionLocal
    try:
        db = SessionLocal()

        # Evaluate
        try:
            from ragas import SingleTurnSample
            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
            )
            eval_scores = await rag_evaluator._async_evaluate(sample)
        except Exception as e:
            eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "error": str(e)[:100]}

        # Log to DB
        query_log = QueryLog(
            query=query,
            response=answer,
            model_used=model_used,
            latency_ms=latency_ms,
            token_count=input_tokens + output_tokens,
            cost=cost,
            retrieval_scores={
                "num_sources": len(context_chunks),
                "top_rerank_score": context_chunks[0].get("rerank_score", 0) if context_chunks else 0,
            },
            evaluation_scores=eval_scores,
            config_version=config_version,
        )
        db.add(query_log)
        db.commit()

        # Trigger optimizer if needed (skip if A/B test is still running)
        query_count = db.query(QueryLog).count()
        if query_count % settings.optimizer_trigger_every == 0:
            # Check if there's an unfinished A/B test
            from app.models.schemas import PipelineConfig
            proposed = db.query(PipelineConfig).filter(PipelineConfig.is_active == False).first()
            if not proposed:
                # No pending A/B test — safe to propose new config
                await optimization_agent.run(db)
            else:
                print(f"[Optimizer] Skipped — A/B test still running for config v{proposed.version}")

        db.close()
    except Exception as e:
        print(f"[Background Eval] Failed: {e}")


@app.post("/query-optimized")
async def query_with_agent(request: AgentQueryRequest, db: Session = Depends(get_db)):
    """Query pipeline with config-driven optimization. Config (top_k etc) is managed by optimizer, not user."""
    tracker = LatencyTracker()

    # Read active config (via A/B test service)
    test_state = get_active_config(db)
    config = test_state["use_config"]
    config_version = test_state["config_version"]

    # Check semantic cache
    from app.pipeline.embedding import embedding_service
    query_embedding = embedding_service.generate_embedding(request.query)

    # Clean expired cache entries (older than 1 hour)
    db.execute(text("DELETE FROM query_cache WHERE created_at < NOW() - INTERVAL '1 hour'"))
    db.commit()

    # Search for similar cached question (cosine similarity > 0.95)
    cache_result = db.execute(
        text("""
            SELECT question, response, 1 - (embedding <=> :embedding) as similarity
            FROM query_cache
            WHERE 1 - (embedding <=> :embedding) > 0.95
            ORDER BY similarity DESC
            LIMIT 1
        """),
        {"embedding": str(query_embedding)},
    ).fetchone()

    if cache_result:
        cached_response = cache_result.response
        cached_response["from_cache"] = True
        cached_response["matched_query"] = cache_result.question
        cached_response["similarity"] = round(cache_result.similarity, 4)
        return cached_response

    # Step 1: Retrieve chunks (top_k from config)
    tracker.start("retrieval")
    context_chunks = hybrid_search(request.query, db, top_k=config["top_k"])
    tracker.end("retrieval")

    if not context_chunks:
        return {
            "question": request.query,
            "answer": "No documents found. Please ingest relevant documents.",
            "sources": [],
        }

    # Step 2: Route query to model (routing_threshold from config)
    tracker.start("routing")
    routing = query_router.classify_complexity(request.query, context_chunks)
    tracker.end("routing")

    # Step 3: Generate answer
    tracker.start("generation")
    llm_response = llm_service.generate_response(
        request.query, context_chunks, model_id=routing["model"]
    )
    tracker.end("generation")

    # Step 4: Calculate cost (fast, no API call)
    query_cost = cost_tracker.calculate_cost(
        model_id=llm_response["model_used"],
        input_tokens=llm_response["input_tokens"],
        output_tokens=llm_response["output_tokens"],
    )

    # Build response immediately (don't wait for eval)
    response = {
        "question": request.query,
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "complexity": routing["complexity"],
        "config_used": config,
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "evaluation": "pending (async)",
        "latency": tracker.get_report(),
        "cost": query_cost,
        "num_sources": len(context_chunks),
        "sources": [
            {
                "content": chunk["content"][:200],
                "rerank_score": chunk.get("rerank_score", 0),
            }
            for chunk in context_chunks
        ],
    }

    # Store in semantic cache
    cache_entry = QueryCache(
        question=request.query,
        embedding=query_embedding,
        response=response,
    )
    db.add(cache_entry)
    db.commit()

    response["from_cache"] = False
    response["config_version"] = config_version
    if test_state.get("ab_status"):
        response["ab_test"] = test_state["ab_status"]
    if test_state.get("ab_result"):
        response["ab_result"] = test_state["ab_result"]

    # Step 5: Fire evaluation + logging in background (non-blocking)
    asyncio.create_task(
        run_eval_and_log_background(
            query=request.query,
            answer=llm_response["answer"],
            contexts=[chunk["content"] for chunk in context_chunks],
            model_used=llm_response["model_used"],
            latency_ms=tracker.get_total(),
            input_tokens=llm_response["input_tokens"],
            output_tokens=llm_response["output_tokens"],
            cost=query_cost["total_cost_usd"],
            context_chunks=context_chunks,
            config_version=config_version,
        )
    )

    return response


@app.get("/optimize")
async def run_optimization(window: int = 20, db: Session = Depends(get_db)):
    """Manually trigger the optimization agent."""
    result = await optimization_agent.run(db, window=window)
    return result


@app.get("/ab-test")
async def get_ab_test_status(db: Session = Depends(get_db)):
    """Check current A/B test status."""
    test_state = ab_test_service.get_test_state(db)
    return test_state


@app.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents."""
    docs = db.query(Document).order_by(Document.ingested_at.desc()).all()
    return {
        "total_documents": len(docs),
        "documents": [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "file_type": doc.file_type,
                "ingested_at": str(doc.ingested_at),
                "chunk_config": doc.chunk_config,
            }
            for doc in docs
        ],
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document, its chunks, and clear query cache."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    filename = doc.filename

    # Delete chunks belonging to this document
    chunks_deleted = db.query(Chunk).filter(Chunk.document_id == document_id).delete()

    # Delete the document
    db.delete(doc)

    # Clear query cache (answers may reference this document's content)
    db.execute(text("TRUNCATE TABLE query_cache"))

    db.commit()

    return {
        "message": f"Document '{filename}' deleted successfully",
        "chunks_deleted": chunks_deleted,
        "cache_cleared": True,
    }


@app.delete("/documents")
async def delete_all_documents(db: Session = Depends(get_db)):
    """Delete ALL documents, chunks, and clear query cache."""
    chunks_deleted = db.query(Chunk).delete()
    docs_deleted = db.query(Document).delete()
    db.execute(text("TRUNCATE TABLE query_cache"))
    db.commit()

    return {
        "message": "All documents deleted",
        "documents_deleted": docs_deleted,
        "chunks_deleted": chunks_deleted,
        "cache_cleared": True,
    }