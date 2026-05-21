from typing import List, Dict
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.models.schemas import Chunk


class BM25Index:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, db: Session):
        """Load all chunks from DB and build BM25 index."""
        all_chunks = db.query(Chunk).all()
        self.chunks = all_chunks

        # Tokenize each chunk's content (simple split by space)
        tokenized_docs = [chunk.content.lower().split() for chunk in all_chunks]

        if tokenized_docs:
            self.index = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using BM25 keyword matching."""
        if self.index is None or not self.chunks:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": str(chunk.id),
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "document_id": str(chunk.document_id),
                    "bm25_score": round(float(scores[idx]), 4),
                })

        return results


bm25_index = BM25Index()