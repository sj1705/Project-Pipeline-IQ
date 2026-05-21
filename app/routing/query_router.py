from typing import List, Dict
from app.config import settings


class QueryRouter:
    def __init__(self):
        self.routing_threshold_low = 0.3
        self.routing_threshold_high = 0.7
        self.models = {
            "simple": settings.llm_model_id,           # Haiku — fast, cheap
            "complex": settings.llm_model_complex,     # Sonnet — accurate, expensive
        }

    def classify_complexity(self, query: str, context_chunks: List[Dict]) -> Dict:
        """Classify query complexity based on multiple signals."""
        signals = {
            "word_count": len(query.split()),
            "has_comparison": self._detect_comparison(query),
            "has_reasoning": self._detect_reasoning(query),
            "multi_hop": self._detect_multi_hop(query),
            "context_chunks_needed": len(context_chunks),
        }

        # Compute complexity score (0.0 = simple, 1.0 = complex)
        score = self._compute_score(signals)

        if score < self.routing_threshold_low:
            complexity = "simple"
        elif score < self.routing_threshold_high:
            complexity = "moderate"
        else:
            complexity = "complex"

        return {
            "complexity": complexity,
            "score": round(score, 3),
            "signals": signals,
            "model": self._get_model(complexity),
        }

    def _compute_score(self, signals: Dict) -> float:
        """Weighted scoring of complexity signals."""
        score = 0.0

        # Long queries tend to be more complex
        if signals["word_count"] > 20:
            score += 0.2
        elif signals["word_count"] > 10:
            score += 0.1

        # Comparison questions need reasoning
        if signals["has_comparison"]:
            score += 0.3

        # Reasoning questions need deeper thinking
        if signals["has_reasoning"]:
            score += 0.25

        # Multi-hop questions are complex
        if signals["multi_hop"]:
            score += 0.3

        # Many context chunks needed = complex topic
        if signals["context_chunks_needed"] > 5:
            score += 0.15
        elif signals["context_chunks_needed"] > 3:
            score += 0.1

        return min(score, 1.0)

    def _detect_comparison(self, query: str) -> bool:
        """Detect if query asks for comparison."""
        comparison_words = ["compare", "difference", "versus", "vs", "better", "worse",
                           "similarities", "contrast", "pros and cons"]
        query_lower = query.lower()
        return any(word in query_lower for word in comparison_words)

    def _detect_reasoning(self, query: str) -> bool:
        """Detect if query requires reasoning/analysis."""
        reasoning_words = ["why", "how does", "explain", "analyze", "evaluate",
                          "what causes", "implications", "impact", "recommend"]
        query_lower = query.lower()
        return any(word in query_lower for word in reasoning_words)

    def _detect_multi_hop(self, query: str) -> bool:
        """Detect if query requires combining info from multiple sources."""
        multi_hop_words = ["and also", "in addition", "across all", "throughout",
                          "combine", "both", "all of the", "summarize everything"]
        query_lower = query.lower()
        return any(word in query_lower for word in multi_hop_words)

    def _get_model(self, complexity: str) -> str:
        """Map complexity to model."""
        if complexity == "complex":
            return self.models["complex"]
        return self.models["simple"]


query_router = QueryRouter()