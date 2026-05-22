from typing import List,Dict
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self,model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """ms-macro-MiniLM-L-6-v2 small 22 MB , trained on search relevance data"""
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Score each (query, chunk) pair and return top_k by relevance."""
        if not chunks:
            return []

        #Create pairs for the model to score
        pairs = [(query, chunk["content"]) for chunk in chunks]

        #Score all pairs (return array of floats)
        scores = self.model.predict(pairs)

        #Attach Scores
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = round(float(scores[i]), 4)

        #Sort by Rerank Score
        reranked=sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        return reranked[:top_k]

reranker = Reranker()