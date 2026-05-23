from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END


# Define the shared state — all nodes read/write to this
class PipelineState(TypedDict):
    query: str
    top_k: int
    chunks: List[Dict]
    answer: str
    model_used: str
    latency: Dict
    cost: Dict
    evaluation: Dict


# Node 1: Retrieval
def retrieval_node(state: PipelineState) -> PipelineState:
    """Retrieve relevant chunks using hybrid search."""
    from app.pipeline.retrieval import hybrid_search
    from app.models.database import SessionLocal

    db = SessionLocal()
    chunks = hybrid_search(state["query"], db, top_k=state["top_k"])
    db.close()

    return {"chunks": chunks}


# Node 2: Generation
def generation_node(state: PipelineState) -> PipelineState:
    """Generate answer using LLM with retrieved context."""
    from app.pipeline.generation import llm_service
    from app.routing.query_router import query_router

    # Route to appropriate model
    routing = query_router.classify_complexity(state["query"], state["chunks"])

    # Generate response
    llm_response = llm_service.generate_response(
        state["query"], state["chunks"], model_id=routing["model"]
    )

    return {
        "answer": llm_response["answer"],
        "model_used": llm_response["model_used"],
    }


# Build the graph
def build_pipeline_graph():
    """Create a simple 2-node RAG pipeline graph."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)

    # Define edges (flow)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile into a runnable
    return graph.compile()


# Create the compiled graph
rag_pipeline = build_pipeline_graph()