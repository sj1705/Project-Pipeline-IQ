from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.pipeline.embedding import embedding_service
from app.pipeline.bm25_search import bm25_index


def vector_search(query: str, db: Session, top_k: int = 5) -> List[Dict]:
    """
    Embed the query and find the most similar chunks in pgvector.
    Uses cosine distance (<=>) for similarity.
    """
    query_embedding = embedding_service.generate_embedding(query)

    sql = text("""
        SELECT
            c.id,
            c.content,
            c.chunk_index,
            c.document_id,
            c.embedding <=> :embedding AS distance
        FROM chunks c
        ORDER BY c.embedding <=> :embedding
        LIMIT :top_k
    """)

    results = db.execute(sql, {"embedding": str(query_embedding), "top_k": top_k}).fetchall()

    chunks = []
    for row in results:
        chunks.append({
            "chunk_id": str(row.id),
            "content": row.content,
            "chunk_index": row.chunk_index,
            "document_id": str(row.document_id),
            "distance": round(row.distance, 4),
            "similarity": round(1 - row.distance, 4),
        })

    return chunks


def hybrid_search(query: str, db: Session, top_k: int = 5) -> List[Dict]:
    """
    Combine vector search (semantic) + BM25 (keyword) using Reciprocal Rank Fusion.
    """
    # Rebuild BM25 index (in production, do this on ingest, not every query)
    bm25_index.build_index(db)

    # Get results from both methods (fetch more than top_k for better fusion)
    vector_results = vector_search(query, db, top_k=top_k * 2)
    bm25_results = bm25_index.search(query, top_k=top_k * 2)

    # Reciprocal Rank Fusion
    rrf_scores = {}  # chunk_id -> score
    chunk_data = {}  # chunk_id -> chunk info

    # Score vector results
    for rank, chunk in enumerate(vector_results):
        chunk_id = chunk["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 60)
        chunk_data[chunk_id] = chunk

    # Score BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 60)
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = chunk

    # Sort by RRF score and return top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results = []
    for chunk_id in sorted_ids:
        chunk = chunk_data[chunk_id]
        chunk["rrf_score"] = round(rrf_scores[chunk_id], 4)
        results.append(chunk)

    return results