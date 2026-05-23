import time
from typing import TypedDict, List, Dict, Literal, Annotated
from langgraph.graph import StateGraph, END
from app.config import settings


class PipelineState(TypedDict):
    query: str
    top_k: int
    chunks: List[Dict]
    answer: str
    model_used: str
    input_tokens: int
    output_tokens: int
    evaluation: Dict
    retry_count: int
    latency_stages: Dict
    cost: Dict
    _stage_start: float  # internal: timing helper


# --- Helper for timing ---
def _time_ms():
    return time.time() * 1000


# --- Node 1: Retrieval ---
def retrieval_node(state: PipelineState) -> dict:
    """Retrieve relevant chunks using hybrid search."""
    from app.pipeline.retrieval import hybrid_search
    from app.models.database import SessionLocal

    start = _time_ms()

    db = SessionLocal()
    chunks = hybrid_search(state["query"], db, top_k=state["top_k"])
    db.close()

    elapsed = round(_time_ms() - start, 2)
    latency = state.get("latency_stages", {})
    latency["retrieval"] = elapsed

    return {"chunks": chunks, "latency_stages": latency}


# --- Node 2: Generation ---
def generation_node(state: PipelineState) -> dict:
    """Generate answer using routed model."""
    from app.pipeline.generation import llm_service
    from app.routing.query_router import query_router

    start = _time_ms()

    routing = query_router.classify_complexity(state["query"], state["chunks"])
    llm_response = llm_service.generate_response(
        state["query"], state["chunks"], model_id=routing["model"]
    )

    elapsed = round(_time_ms() - start, 2)
    latency = state.get("latency_stages", {})
    latency["generation"] = elapsed

    return {
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "latency_stages": latency,
    }


# --- Node 3: Evaluation ---
def evaluation_node(state: PipelineState) -> dict:
    """Evaluate the generated answer for quality."""
    from app.evaluation.ragas_eval import rag_evaluator
    from ragas import SingleTurnSample
    import asyncio

    start = _time_ms()

    contexts = [chunk["content"] for chunk in state["chunks"]]
    sample = SingleTurnSample(
        user_input=state["query"],
        response=state["answer"],
        retrieved_contexts=contexts,
    )

    loop = asyncio.get_event_loop()
    scores = loop.run_until_complete(rag_evaluator._async_evaluate(sample))

    elapsed = round(_time_ms() - start, 2)
    latency = state.get("latency_stages", {})
    latency["evaluation"] = elapsed

    return {"evaluation": scores, "latency_stages": latency}


# --- Node 4: Retry with Sonnet ---
def retry_with_sonnet_node(state: PipelineState) -> dict:
    """Retry generation with the more powerful model."""
    from app.pipeline.generation import llm_service

    start = _time_ms()

    llm_response = llm_service.generate_response(
        state["query"], state["chunks"], model_id=settings.llm_model_complex
    )

    elapsed = round(_time_ms() - start, 2)
    latency = state.get("latency_stages", {})
    latency["retry_generation"] = elapsed

    return {
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
        "input_tokens": llm_response["input_tokens"],
        "output_tokens": llm_response["output_tokens"],
        "retry_count": state["retry_count"] + 1,
        "latency_stages": latency,
    }


# --- Node 5: Calculate Cost ---
def cost_node(state: PipelineState) -> dict:
    """Calculate the cost of this query."""
    from app.evaluation.cost_tracker import cost_tracker

    query_cost = cost_tracker.calculate_cost(
        model_id=state["model_used"],
        input_tokens=state["input_tokens"],
        output_tokens=state["output_tokens"],
    )

    return {"cost": query_cost}


# --- Node 6: Log to Database ---
def log_to_db_node(state: PipelineState) -> dict:
    """Store query metrics in the database."""
    from app.models.database import SessionLocal
    from app.models.schemas import QueryLog

    db = SessionLocal()

    total_latency = sum(state.get("latency_stages", {}).values())

    query_log = QueryLog(
        query=state["query"],
        response=state["answer"],
        model_used=state["model_used"],
        latency_ms=round(total_latency, 2),
        token_count=state["input_tokens"] + state["output_tokens"],
        cost=state.get("cost", {}).get("total_cost_usd", 0),
        retrieval_scores={
            "num_sources": len(state["chunks"]),
            "top_rerank_score": state["chunks"][0].get("rerank_score", 0) if state["chunks"] else 0,
        },
        evaluation_scores=state.get("evaluation", {}),
    )
    db.add(query_log)
    db.commit()
    db.close()

    return {}


# --- Conditional: should we retry? ---
def should_retry(state: PipelineState) -> Literal["retry", "finish"]:
    """Decide if we need to retry with a better model."""
    eval_scores = state.get("evaluation", {})
    faithfulness = eval_scores.get("faithfulness", 1.0)

    if faithfulness < 0.7 and state["retry_count"] == 0:
        return "retry"
    return "finish"


# --- Build the full graph ---
def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("retry_sonnet", retry_with_sonnet_node)
    graph.add_node("calculate_cost", cost_node)
    graph.add_node("log_to_db", log_to_db_node)

    # Define flow
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    # Conditional: after evaluation
    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "retry_sonnet",
            "finish": "calculate_cost",
        }
    )

    # After retry → calculate cost
    graph.add_edge("retry_sonnet", "calculate_cost")

    # After cost → log to db → end
    graph.add_edge("calculate_cost", "log_to_db")
    graph.add_edge("log_to_db", END)

    return graph.compile()


rag_pipeline = build_pipeline_graph()