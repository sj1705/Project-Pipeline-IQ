import json
import boto3
from typing import List, Dict
from app.config import settings


class LLMService:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    def generate_response(self, question: str, context_chunks: List[Dict], model_id: str = None) -> Dict:
        """Generate a response using retrieved context and an LLM."""
        if model_id is None:
            model_id = settings.llm_model_id

        # Build Context for LLM
        context = "\n\n---\n\n".join([chunk["content"] for chunk in context_chunks])

        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

        # Call Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        })

        response = self.client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]

        # Get token usage
        usage = result.get("usage", {})

        return {
            "answer": answer,
            "model_used": model_id,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }


llm_service = LLMService()
