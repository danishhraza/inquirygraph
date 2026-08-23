# InquiryGraph

AI Research & Investigation Agent built with **LangGraph**, **GraphRAG** (Neo4j), and **hybrid retrieval** (Qdrant + BM25).

## What works today

- 8-node LangGraph workflow with a real research loop
- Web search (DuckDuckGo free, or Tavily if configured) with SSRF protection
- Chunk → embed → store in **Qdrant** (vector RAG)
- Extract entities/claims → store in **Neo4j** (knowledge graph)
- Retrieval modes: `vector`, `hybrid`, `graph`, `graph_plus_vector`
- Local cross-encoder reranking (optional)
- Contradiction detection → `CONTRADICTS` edges in Neo4j
- Citation verification pass (anti-hallucination)
- Structured final report via FastAPI
- Local `.txt`/`.md`/`.pdf` document upload and indexing
- Investigation lookup via `GET /investigations/{id}`
- Sync or async job processing (`PROCESSING_MODE`)
- Minimal web frontend at `/`
- MCP tool server (`scripts/run_mcp.py`)
- Reproducible benchmark under `eval/`

## Quick start

```bash
# 1. Start databases
docker compose up -d

# 2. Configure secrets
copy .env.example .env
# Add OPENROUTER_API_KEY to .env

# 3. Install
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# 4. Run API
python scripts/run_api.py
```

## Run an investigation

```bash
curl -X POST http://localhost:8000/investigations ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Compare Databricks, Snowflake, and BigQuery for on-prem migration.\"}"
```

Or open the frontend at http://localhost:8000 or the Swagger docs at http://localhost:8000/docs

## LangGraph flow (8 nodes)

```
understand_query
      ↓
plan_research
      ↓
gather_and_index ──┐
      ↓            │
retrieve_evidence  │  research loop
      ↓            │
check_coverage ────┘  (adds follow-up tasks if evidence is thin)
      ↓
detect_contradictions   (LLM + Neo4j CONTRADICTS edges)
      ↓
synthesize_report       (uses evidence + detected contradictions)
      ↓
verify_citations        (checks report claims against citations)
      ↓
     END
```

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | LangGraph | Stateful agent workflow + checkpointing |
| LLM | OpenRouter (DeepSeek V4 Flash + Qwen 30B) | Plan, extract, synthesize |
| Embeddings | FastEmbed (local, free) | Convert text → vectors |
| Vector DB | Qdrant | Similarity search (RAG) |
| Knowledge Graph | Neo4j | Entities, claims, relationships |
| API | FastAPI | HTTP interface |
| Frontend | Static HTML/JS | Minimal demo UI |
| Tools | MCP (optional) | Expose tools to MCP clients |

## MCP tool server

Expose `web_search` and `fetch_url` to any MCP client (Cursor, Claude Desktop):

```bash
pip install -e ".[mcp]"
python scripts/run_mcp.py
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | **Required** LLM API key |
| `LLM_MAIN_MODEL` | `deepseek/deepseek-v4-flash-0731` | Planning/synthesis |
| `LLM_FAST_MODEL` | `qwen/qwen3-30b-a3b-instruct-2507` | Extraction/verification |
| `RETRIEVAL_STRATEGY` | `hybrid` | vector/hybrid/graph/graph_plus_vector |
| `PROCESSING_MODE` | `sync` | `sync` or `async` |
| `TAVILY_API_KEY` | — | Optional better search |

## Not built yet

- Production async message queue (Redis/Celery), authentication, and rate limits
- Larger curated benchmark corpus with gold citations
