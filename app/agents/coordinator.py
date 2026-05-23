from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
from app.config import settings


class PipelineState(TypedDict):
    query: str
    top_k: int
    chunks: List[Dict]
    answer: str
    model_used: str
    evaluation: Dict
    retry_count: int


# Node 1: Retrieval
def retrieval_node(state: PipelineState) -> dict:
    """Retrieve relevant chunks using hybrid search."""
    from app.pipeline.retrieval import hybrid_search
    from app.models.database import SessionLocal

    db = SessionLocal()
    chunks = hybrid_search(state["query"], db, top_k=state["top_k"])
    db.close()

    return {"chunks": chunks}


# Node 2: Generation
def generation_node(state: PipelineState) -> dict:
    """Generate answer using routed model."""
    from app.pipeline.generation import llm_service
    from app.routing.query_router import query_router

    routing = query_router.classify_complexity(state["query"], state["chunks"])
    llm_response = llm_service.generate_response(
        state["query"], state["chunks"], model_id=routing["model"]
    )

    return {
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
    }


# Node 3: Evaluation
def evaluation_node(state: PipelineState) -> dict:
    """Evaluate the generated answer for quality."""
    from app.evaluation.ragas_eval import rag_evaluator
    from ragas import SingleTurnSample
    import asyncio

    contexts = [chunk["content"] for chunk in state["chunks"]]
    sample = SingleTurnSample(
        user_input=state["query"],
        response=state["answer"],
        retrieved_contexts=contexts,
    )

    loop = asyncio.get_event_loop()
    scores = loop.run_until_complete(rag_evaluator._async_evaluate(sample))

    return {"evaluation": scores}


# Node 4: Retry with Sonnet (escalation)
def retry_with_sonnet_node(state: PipelineState) -> dict:
    """Retry generation with the more powerful model."""
    from app.pipeline.generation import llm_service

    llm_response = llm_service.generate_response(
        state["query"], state["chunks"], model_id=settings.llm_model_complex
    )

    return {
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "retry_count": state["retry_count"] + 1,
    }


# Conditional edge: should we retry?
def should_retry(state: PipelineState) -> Literal["retry", "finish"]:
    """Decide if we need to retry with a better model."""
    eval_scores = state.get("evaluation", {})
    faithfulness = eval_scores.get("faithfulness", 1.0)

    # If faithfulness is low AND we haven't retried yet → retry
    if faithfulness < 0.7 and state["retry_count"] == 0:
        return "retry"
    return "finish"


# Build the full graph
def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("retry_sonnet", retry_with_sonnet_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    # Conditional: after evaluation, decide retry or finish
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "retry_sonnet",
            "finish": END,
        }
    )

    # After retry, go straight to END (don't evaluate again to avoid infinite loop)
    graph.add_edge("retry_sonnet", END)

    return graph.compile()


rag_pipeline = build_pipeline_graph()