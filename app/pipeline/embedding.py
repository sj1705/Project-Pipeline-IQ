import json
import boto3
from typing import List
from app.config import settings


class EmbeddingService:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.model_id = settings.embedding_model_id

    def generate_embedding(self,text: str) -> List[float]:
        """Generate embeddings for the single text chunk using AWS Bedrock."""
        body=json.dumps({"inputText": text})
        response= self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result=json.loads(response["body"].read())
        return result.get("embedding", [])

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text chunks."""
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


embedding_service = EmbeddingService()