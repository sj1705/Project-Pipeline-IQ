# PipelineIQ — Self-Optimizing RAG Orchestration System

A production-style RAG (Retrieval-Augmented Generation) pipeline that monitors its own performance and automatically tunes retrieval parameters using an LLM-powered optimization agent.

## What Makes This Different

Most RAG systems are static — you set chunk_size, top_k, and hope for the best. PipelineIQ **optimizes itself**:

1. Every query is evaluated for quality (RAGAS), timed for latency, and tracked for cost
2. Every 20 queries, an LLM-powered optimizer agent analyzes trends
3. The optimizer proposes config changes (different model routing? different top_k?)
4. Changes are A/B tested against the current config
5. Winner is auto-promoted — the pipeline gets better over time

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Semantic Cache (pgvector)                                  │
│  Similar question asked before? → Return cached response    │
└──────────────────────┬──────────────────────────────────────┘
                       ▼ (cache miss)
┌─────────────────────────────────────────────────────────────┐
│  Pipeline (config-driven)                                   │
│  1. Read active config from DB (top_k, rerank_weight, etc)  │
│  2. Hybrid Search: Vector + BM25 + RRF + Cross-encoder      │
│  3. Route to Haiku/Sonnet based on query complexity         │
│  4. Generate answer (AWS Bedrock)                           │
│  5. Evaluate quality (RAGAS: faithfulness, relevancy)       │
│  6. Track latency + cost                                    │
│  7. Log everything to query_logs                            │
└──────────────────────┬──────────────────────────────────────┘
                       ▼ (every 20 queries)
┌─────────────────────────────────────────────────────────────┐
│  Optimizer Agent (LangGraph + Haiku)                        │
│  - Reads metrics from query_logs                            │
│  - Reads its own past decisions from pipeline_configs       │
│  - Reasons about quality/speed/cost tradeoffs               │
│  - Proposes new config → A/B tested → winner promoted       │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API | FastAPI | Async, fast, auto-docs |
| Database | PostgreSQL 16 + pgvector | Vector search + relational in one DB |
| Vector Search | pgvector (cosine distance) | No extra service needed |
| Keyword Search | BM25 (rank-bm25) | Complements vector search for exact terms |
| Reranking | Cross-encoder (sentence-transformers) | Better relevance than vector alone |
| Embeddings | AWS Bedrock Titan Embed v2 (1024 dims) | Production-grade, scalable |
| LLM | AWS Bedrock Claude 3 Haiku / Sonnet | Fast (Haiku) + Smart (Sonnet) |
| Evaluation | RAGAS v0.4 | Industry standard RAG evaluation |
| Optimization Agent | LangGraph + tool use | Real agent — LLM decides, not if-else |
| A/B Testing | Custom (PostgreSQL-based) | Test configs before promoting |
| Semantic Cache | pgvector similarity search | Skip pipeline for similar questions |
| Tracing | LangSmith | Full visibility into agent reasoning |
| Dashboard | Streamlit | Real-time metrics visualization |
| Containerization | Docker Compose | One command to start everything |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health (DB status) |
| POST | `/ingest` | Upload PDF/DOCX/HTML → chunk → embed → store |
| POST | `/search` | Vector similarity search (raw) |
| POST | `/query` | Full RAG pipeline (legacy, with RAGAS eval) |
| POST | `/query-optimized` | Config-driven pipeline + semantic cache + A/B testing |
| GET | `/optimize` | Manually trigger optimization agent |
| GET | `/ab-test` | Check A/B test status |
| GET | `/metrics` | View recent query logs + metrics |

## Optimization Agent

The optimizer is a **real LangGraph agent** (not hardcoded rules). It uses Claude Haiku with 3 tools:

- `read_query_metrics()` — Performance stats from last N queries
- `read_past_configs()` — Its own optimization history (avoids repeating mistakes)
- `propose_config()` — Save new config values to DB

### Tunable Parameters (instant-apply, no re-ingestion)

| Parameter | Range | Effect |
|-----------|-------|--------|
| top_k | 3-10 | Chunks retrieved (more = better context, slower) |
| rerank_weight | 0.3-0.9 | Cross-encoder trust (higher = better chunk selection) |
| routing_threshold | 0.3-0.8 | Higher = more queries go to cheap/fast Haiku |
| retry_threshold | 0.5-0.85 | Lower = retry more aggressively with Sonnet |

### Why Not Tune chunk_size?

Chunk size requires re-ingestion of all documents — it can't be applied instantly. The optimizer only controls parameters that take effect on the next query.

## A/B Testing

When the optimizer proposes a new config:
1. Queries alternate between current config and proposed config
2. After 10 queries each → compare avg faithfulness + latency
3. Winner is auto-promoted to active
4. Loser is discarded

This prevents bad configs from being applied blindly.

## Semantic Cache

Uses pgvector to find semantically similar previously-asked questions:
- Similarity > 0.95 → return cached response (~200ms vs ~5000ms)
- New document ingested → entire cache invalidated (answers may be stale)
- No Redis needed — PostgreSQL handles everything

## Setup

### Prerequisites
- Python 3.11
- Docker (for PostgreSQL + pgvector)
- AWS account with Bedrock access (ap-south-1)

### Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/sj1705/Project-Pipeline-IQ.git
cd pipelineiq

# 2. Virtual environment
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start PostgreSQL with pgvector
docker run -d --name pipelineiq-db -e POSTGRES_PASSWORD=postgres123 -e POSTGRES_DB=pipelineiq -p 5433:5432 pgvector/pgvector:pg16

# 5. Configure .env (copy from example and fill in AWS creds)
cp .env.example .env

# 6. Run API
python -m uvicorn app.main:app --reload --port 8003

# 7. Run Dashboard (separate terminal)
streamlit run dashboard/app.py

# 8. Seed sample data
python scripts/seed.py
```

### Quick Start (Docker Compose)

```bash
# Start everything (API + DB + Dashboard)
docker-compose up --build

# Access:
# API: http://localhost:8003
# Dashboard: http://localhost:8501
# Swagger docs: http://localhost:8003/docs
```

### Environment Variables (.env)

```
DATABASE_URL=postgresql+psycopg://postgres:postgres123@localhost:5433/pipelineiq
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
LLM_MODEL_COMPLEX=anthropic.claude-3-sonnet-20240229-v1:0
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=Project-Pipeline-IQ
```

## Project Structure

```
pipelineiq/
├── app/
│   ├── main.py                  # FastAPI app + all endpoints
│   ├── config.py                # Pydantic settings
│   ├── agents/
│   │   └── optimizer.py         # LangGraph optimization agent (real agent with tools)
│   ├── models/
│   │   ├── database.py          # SQLAlchemy engine + session
│   │   └── schemas.py           # ORM models (Document, Chunk, QueryLog, PipelineConfig, QueryCache)
│   ├── pipeline/
│   │   ├── ingestion.py         # PDF/DOCX/HTML parsing (PyMuPDF, python-docx, BeautifulSoup)
│   │   ├── chunking.py          # Recursive text chunking (configurable size/overlap)
│   │   ├── embedding.py         # AWS Titan Embed v2 (1024 dims)
│   │   ├── retrieval.py         # Vector + hybrid search (pgvector + BM25 + RRF)
│   │   ├── bm25_search.py       # BM25 keyword search index
│   │   ├── reranker.py          # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│   │   └── generation.py        # Bedrock LLM response generation
│   ├── routing/
│   │   └── query_router.py      # Query complexity classifier (simple/moderate/complex)
│   ├── evaluation/
│   │   ├── ragas_eval.py        # RAGAS v0.4 (faithfulness, relevancy, context precision)
│   │   ├── latency_tracker.py   # Per-stage timing
│   │   └── cost_tracker.py      # Token cost calculation per model
│   └── services/
│       ├── storage_service.py   # Local file storage
│       └── ab_test_service.py   # A/B testing logic (alternate, compare, promote)
├── dashboard/
│   └── app.py                   # Streamlit dashboard (5 pages)
├── scripts/
│   └── seed.py                  # Seed script (ingest docs + run sample queries)
├── uploads/                     # Uploaded documents stored here
├── docker-compose.yml           # Full stack: API + DB + Dashboard
├── Dockerfile                   # API container
├── Dockerfile.dashboard         # Dashboard container
├── .dockerignore
├── requirements.txt
└── README.md
```

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| pgvector over FAISS/Pinecone | Already need Postgres, free, no extra service |
| pgvector semantic cache handles caching in existing DB |
| LangGraph for optimizer | Real agent: LLM decides tool order, not hardcoded |
| A/B test before promoting | Prevents bad configs from hurting all queries |
| Haiku as optimizer brain | Cheap (~$0.001/run), fast, good enough for config reasoning |
| DB as communication layer | Pipeline writes logs, optimizer reads them — fully decoupled |

## What I Learned Building This

1. **LangGraph is for decision-making, not pipelines** — If your flow is fixed (A → B → C), you don't need a graph
2. **Chunk size is a design decision, not a tunable** — It requires re-ingestion, so optimize retrieval-time params instead
3. **Semantic cache has tradeoffs** — Similar questions might get wrong cached answers; threshold of 0.95 keeps it safe
4. **A/B testing is essential** — Never blindly apply optimizer suggestions
5. **The optimizer is just 3 numbers** — But those 3 numbers (top_k, rerank_weight, routing_threshold) control the quality/speed/cost triangle
