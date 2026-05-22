import asyncio
import nest_asyncio
from typing import Dict, List
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrock, BedrockEmbeddings
from app.config import settings

# Allow nested event loops
nest_asyncio.apply()


class RAGEvaluator:
    """
    RAG evaluator using RAGAS v0.4 with Bedrock as the judge LLM.
    """

    def __init__(self):
        # Set up Bedrock LLM for RAGAS
        bedrock_llm = ChatBedrock(
            model_id=settings.llm_model_id,
            region_name=settings.aws_region,
            credentials_profile_name=None,
            model_kwargs={"temperature": 0},
        )
        self.evaluator_llm = LangchainLLMWrapper(bedrock_llm)

        # Set up Bedrock Embeddings for RAGAS
        bedrock_embeddings = BedrockEmbeddings(
            model_id=settings.embedding_model_id,
            region_name=settings.aws_region,
            credentials_profile_name=None,
        )
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

        # Initialize metrics
        self.faithfulness = Faithfulness(llm=self.evaluator_llm)
        self.relevancy = ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.evaluator_embeddings)
        self.context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate a RAG response using RAGAS metrics.
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        try:
            loop = asyncio.get_event_loop()
            scores = loop.run_until_complete(self._async_evaluate(sample))
            return scores
        except Exception as e:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "error": str(e),
            }

    async def _async_evaluate(self, sample: SingleTurnSample) -> Dict[str, float]:
        """Run all RAGAS metrics asynchronously."""
        faithfulness_score = await self.faithfulness.single_turn_ascore(sample)
        relevancy_score = await self.relevancy.single_turn_ascore(sample)
        precision_score = await self.context_precision.single_turn_ascore(sample)

        return {
            "faithfulness": round(float(faithfulness_score), 4),
            "answer_relevancy": round(float(relevancy_score), 4),
            "context_precision": round(float(precision_score), 4),
        }


rag_evaluator = RAGEvaluator()
