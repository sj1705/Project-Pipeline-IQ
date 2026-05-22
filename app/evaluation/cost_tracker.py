from typing import Dict


# Pricing per 1M tokens (USD) — update these if prices change
MODEL_PRICING = {
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input": 0.25,
        "output": 1.25,
    },
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input": 3.00,
        "output": 15.00,
    },
    "anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 3.00,
        "output": 15.00,
    },
    "amazon.titan-embed-text-v2:0": {
        "input": 0.02,
        "output": 0.0,
    },
}


class CostTracker:
    """Calculate cost per query based on token usage and model pricing."""

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict:
        """
        Calculate the cost of a single LLM call.
        Returns cost breakdown in USD.
        """
        pricing = MODEL_PRICING.get(model_id, {"input": 0.0, "output": 0.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "model": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }

    def calculate_embedding_cost(self, token_count: int) -> Dict:
        """Calculate cost of embedding generation."""
        pricing = MODEL_PRICING["amazon.titan-embed-text-v2:0"]
        cost = (token_count / 1_000_000) * pricing["input"]

        return {
            "embedding_tokens": token_count,
            "embedding_cost_usd": round(cost, 6),
        }

    def calculate_query_total_cost(
        self,
        generation_model: str,
        generation_input_tokens: int,
        generation_output_tokens: int,
        embedding_tokens: int,
        evaluation_input_tokens: int = 0,
        evaluation_output_tokens: int = 0,
    ) -> Dict:
        """Calculate total cost of a full query (embedding + generation + evaluation)."""
        gen_cost = self.calculate_cost(generation_model, generation_input_tokens, generation_output_tokens)
        embed_cost = self.calculate_embedding_cost(embedding_tokens)

        # Evaluation uses Haiku (cheap)
        eval_cost = self.calculate_cost(
            "anthropic.claude-3-haiku-20240307-v1:0",
            evaluation_input_tokens,
            evaluation_output_tokens,
        )

        total = gen_cost["total_cost_usd"] + embed_cost["embedding_cost_usd"] + eval_cost["total_cost_usd"]

        return {
            "generation": gen_cost,
            "embedding": embed_cost,
            "evaluation_cost_usd": eval_cost["total_cost_usd"],
            "total_query_cost_usd": round(total, 6),
        }


cost_tracker = CostTracker()