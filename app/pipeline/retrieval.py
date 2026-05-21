from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.pipeline.embedding import embedding_service


def vector_search(query: str, db: Session, top_k: int = 5) -> List[Dict]:
    """
    Embed the query and find the most similar chunks in pgvector.
    Uses cosine distance (<=>) for similarity.
    """
    # Step 1: Embed the query
    query_embedding = embedding_service.generate_embedding(query)

    # Step 2: Search pgvector using cosine distance
    # <=> is pgvector's cosine distance operator (lower = more similar)
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

    # Step 3: Format results
    chunks = []
    for row in results:
        chunks.append({
            "chunk_id": str(row.id),
            "content": row.content,
            "chunk_index": row.chunk_index,
            "document_id": str(row.document_id),
            "distance": round(row.distance, 4),
            "similarity": round(1 - row.distance, 4),  # convert distance to similarity score
        })

    return chunks