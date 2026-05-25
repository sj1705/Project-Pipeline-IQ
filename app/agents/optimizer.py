"""
Optimization Agent — Real LangGraph agent with tools.

This agent uses an LLM (Haiku) to reason about pipeline performance
and decide which config changes to make. It has tools to:
1. Read recent query metrics from DB
2. Read its own past optimization decisions
3. Propose new config values

The LLM DECIDES which tools to call and in what order.
This is a REAL agent — not hardcoded if-else rules.
"""

import json
from typing import Dict, List, TypedDict, Annotated
from sqlalchemy.orm import Session
from sqlalchemy import desc
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_aws import ChatBedrock
from app.config import settings
from app.models.schemas import QueryLog, PipelineConfig


# --- Agent State ---
class OptimizerState(TypedDict):
    messages: List  # conversation history (LLM + tool calls)
    db: Session  # database session (passed through state)
    result: Dict  # final optimization result


# --- Tools the agent can use ---

@tool
def read_query_metrics(window: int = 20) -> str:
    """
    Read performance metrics from the last N queries.
    Returns: avg faithfulness, avg latency, avg cost, model distribution.
    Use this to understand how the pipeline is currently performing.
    """
    # DB session is injected via closure (see OptimizerAgent.run)
    from app.models.database import SessionLocal
    db = SessionLocal()

    recent_logs = (
        db.query(QueryLog)
        .order_by(desc(QueryLog.created_at))
        .limit(window)
        .all()
    )

    if not recent_logs:
        db.close()
        return json.dumps({"status": "no_data", "message": "No query logs found"})

    total = len(recent_logs)
    avg_latency = sum(log.latency_ms or 0 for log in recent_logs) / total
    avg_cost = sum(log.cost or 0 for log in recent_logs) / total

    faithfulness_scores = []
    relevancy_scores = []
    for log in recent_logs:
        if log.evaluation_scores:
            if "faithfulness" in log.evaluation_scores:
                faithfulness_scores.append(log.evaluation_scores["faithfulness"])
            if "answer_relevancy" in log.evaluation_scores:
                relevancy_scores.append(log.evaluation_scores["answer_relevancy"])

    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores else 0.0
    )
    avg_relevancy = (
        sum(relevancy_scores) / len(relevancy_scores)
        if relevancy_scores else 0.0
    )

    model_counts = {}
    for log in recent_logs:
        model = log.model_used or "unknown"
        model_counts[model] = model_counts.get(model, 0) + 1

    db.close()

    return json.dumps({
        "total_queries_analyzed": total,
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_relevancy": round(avg_relevancy, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_cost_usd": round(avg_cost, 6),
        "model_distribution": model_counts,
    })


@tool
def read_past_configs(limit: int = 5) -> str:
    """
    Read the last N pipeline config versions (including current active one).
    Shows what config changes were made previously and which is currently active.
    Use this to see your own optimization history and avoid repeating failed changes.
    """
    from app.models.database import SessionLocal
    db = SessionLocal()

    configs = (
        db.query(PipelineConfig)
        .order_by(desc(PipelineConfig.version))
        .limit(limit)
        .all()
    )

    if not configs:
        db.close()
        return json.dumps({"status": "no_configs", "message": "No configs found. Using defaults: top_k=5, rerank_weight=0.5, routing_threshold=0.5, retry_threshold=0.7"})

    result = []
    for cfg in configs:
        result.append({
            "version": cfg.version,
            "top_k": cfg.top_k,
            "rerank_weight": cfg.rerank_weight,
            "routing_threshold": cfg.routing_threshold,
            "retry_threshold": cfg.retry_threshold,
            "is_active": cfg.is_active,
            "created_at": str(cfg.created_at),
        })

    db.close()
    return json.dumps(result)


@tool
def propose_config(top_k: int, rerank_weight: float, routing_threshold: float, retry_threshold: float, reasoning: str) -> str:
    """
    Propose a new pipeline configuration. This saves it to the database as inactive.
    It will be A/B tested before being promoted to active.

    Parameters:
    - top_k (3-10): Number of chunks to retrieve. More = better context but slower.
    - rerank_weight (0.3-0.9): Trust in cross-encoder reranking. Higher = better selection.
    - routing_threshold (0.3-0.8): Score above this sends query to expensive model. Higher = more queries go to cheap model.
    - retry_threshold (0.5-0.85): Faithfulness below this triggers retry with stronger model.
    - reasoning: Explain WHY you're making these changes.
    """
    from app.models.database import SessionLocal
    db = SessionLocal()

    # Clamp values to bounds
    top_k = max(3, min(10, top_k))
    rerank_weight = max(0.3, min(0.9, round(rerank_weight, 1)))
    routing_threshold = max(0.3, min(0.8, round(routing_threshold, 1)))
    retry_threshold = max(0.5, min(0.85, round(retry_threshold, 2)))

    # Get next version number
    latest = db.query(PipelineConfig).order_by(desc(PipelineConfig.version)).first()
    next_version = (latest.version + 1) if latest else 1

    new_config = PipelineConfig(
        version=next_version,
        chunk_size=512,  # Fixed — not tunable
        chunk_overlap=50,  # Fixed — not tunable
        top_k=top_k,
        rerank_weight=rerank_weight,
        routing_threshold=routing_threshold,
        retry_threshold=retry_threshold,
        is_active=False,  # Proposed only — needs A/B test to promote
    )
    db.add(new_config)
    db.commit()
    db.close()

    return json.dumps({
        "status": "config_proposed",
        "version": next_version,
        "config": {
            "top_k": top_k,
            "rerank_weight": rerank_weight,
            "routing_threshold": routing_threshold,
            "retry_threshold": retry_threshold,
        },
        "reasoning": reasoning,
        "note": "Config saved as INACTIVE. Will be A/B tested before promotion.",
    })


# --- The Agent ---

OPTIMIZER_SYSTEM_PROMPT = """You are a RAG pipeline optimization agent. Your job is to analyze
pipeline performance metrics and propose configuration changes to improve quality, speed, and cost.

You have 3 tools:
1. read_query_metrics — Get performance stats from recent queries
2. read_past_configs — See what configs have been tried before
3. propose_config — Save a new configuration proposal

TUNABLE PARAMETERS (and their effects):
- top_k (3-10): Chunks retrieved. More = better context but higher latency.
- rerank_weight (0.3-0.9): Cross-encoder trust. Higher = better chunk selection.
- routing_threshold (0.3-0.8): Higher = more queries go to cheap/fast model (Haiku).
- retry_threshold (0.5-0.85): Lower = retry more aggressively with expensive model.

OPTIMIZATION RULES:
- If avg faithfulness < 0.7: Pipeline is giving unreliable answers. Try more chunks (top_k) or better ranking (rerank_weight).
- If avg latency > 5000ms: Pipeline is too slow. Reduce top_k or raise routing_threshold.
- If avg cost > $0.05/query: Too expensive. Raise routing_threshold (use Haiku more).
- If you increased a param last time and it didn't help: DON'T increase it again. Try something else.
- CONFLICT (low quality + high latency): Don't change top_k. Rely on rerank_weight.

ALWAYS:
1. First read metrics to understand current state
2. Then read past configs to see what was tried before
3. Then decide: propose a change OR report that no changes are needed
4. In your reasoning, explain the tradeoff you're making

If metrics are all within acceptable bounds, do NOT propose changes. Report that the pipeline is healthy."""


class OptimizerAgent:
    """
    LangGraph-based optimization agent.
    Uses Haiku as the reasoning LLM with tools for DB access.
    """

    def __init__(self):
        # LLM for the agent (cheap + fast)
        self.llm = ChatBedrock(
            model_id=settings.llm_model_complex,  # Haiku
            region_name=settings.aws_region,
            model_kwargs={"temperature": 0},
        )

        # Bind tools to the LLM
        self.tools = [read_query_metrics, read_past_configs, propose_config]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _build_graph(self):
        """Build the LangGraph agent graph."""

        def agent_node(state: OptimizerState) -> dict:
            """The agent thinks and decides which tool to call."""
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": messages + [response]}

        def tool_node(state: OptimizerState) -> dict:
            """Execute the tool the agent requested."""
            messages = state["messages"]
            last_message = messages[-1]

            tool_results = []
            for tool_call in last_message.tool_calls:
                # Find and execute the tool
                tool_fn = {t.name: t for t in self.tools}[tool_call["name"]]
                result = tool_fn.invoke(tool_call["args"])
                tool_results.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

            return {"messages": messages + tool_results}

        def should_continue(state: OptimizerState) -> str:
            """Check if the agent wants to call more tools or is done."""
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Build graph
        graph = StateGraph(OptimizerState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def run(self, db: Session, window: int = 20) -> Dict:
        """Run the optimization agent."""
        graph = self._build_graph()

        # Start the agent with system prompt + trigger message
        initial_messages = [
            SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Analyze the last {window} queries and decide if any config changes are needed. "
                f"Use your tools to read metrics and past configs before making a decision."
            ),
        ]

        # Run the graph
        result = graph.invoke({
            "messages": initial_messages,
            "db": None,  # Not used directly in state
            "result": {},
        })

        # Extract the final response from the agent
        final_message = result["messages"][-1]

        # Parse the conversation to build a structured response
        response = {
            "status": "completed",
            "agent_reasoning": final_message.content if hasattr(final_message, "content") else str(final_message),
            "steps_taken": len(result["messages"]) - 2,  # minus system + initial
        }

        # Check if a config was proposed (look for propose_config tool call in messages)
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and "config_proposed" in msg.content:
                response["optimization"] = json.loads(msg.content)
                response["status"] = "optimization_proposed"
                break
        else:
            response["status"] = "no_changes_needed"

        return response


# Singleton
optimization_agent = OptimizerAgent()
