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
- MCP server that runs full investigations from Claude Desktop, Claude Code or Cursor (`scripts/run_mcp.py`)
- Offline, reproducible retrieval benchmark (recall@k, MRR, config hashing) under `eval/`

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
| Tools | MCP (optional) | Run investigations from any MCP client |

## MCP server

Run InquiryGraph from any MCP client (Claude Desktop, Claude Code, Cursor). The
client gets the whole research pipeline as tools, not just web search:

| Tool | What it does |
|------|--------------|
| `start_investigation(query)` | Starts an investigation in the background and returns its id at once |
| `get_investigation(investigation_id)` | Status and current graph step while running; the cited Markdown report when done |
| `run_investigation(query)` | Runs an investigation and waits for the report, sending a progress notification per graph step |
| `web_search(query, max_results)` | Raw web search |
| `fetch_url(url)` | Readable text of a web page |

The report is also available as the resource `investigation://{id}/report`.
Investigations take a few minutes, longer than many clients wait on one tool
call, so clients should prefer `start_investigation` and poll. Both paths use the
same background job runner as the REST API, so investigations started over MCP
also appear in the API and web UI.

Setup (Docker services and `.env` as in Quick start):

```bash
pip install -e ".[mcp]"
```

Claude Code:

```bash
claude mcp add inquirygraph -- "<repo>/.venv/Scripts/python.exe" "<repo>/scripts/run_mcp.py"
```

Claude Desktop (`claude_desktop_config.json`) or Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "inquirygraph": {
      "command": "<repo>/.venv/Scripts/python.exe",
      "args": ["<repo>/scripts/run_mcp.py"]
    }
  }
}
```

The server speaks MCP over stdio, so all logging goes to stderr.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | **Required** LLM API key |
| `LLM_MAIN_MODEL` | `deepseek/deepseek-v4-flash-0731` | Planning/synthesis |
| `LLM_FAST_MODEL` | `qwen/qwen3-30b-a3b-instruct-2507` | Extraction/verification |
| `RETRIEVAL_STRATEGY` | `hybrid` | vector/hybrid/graph/graph_plus_vector |
| `PROCESSING_MODE` | `sync` | `sync` or `async` |
| `TAVILY_API_KEY` | — | Optional better search |

## Evaluation

`eval/retrieval_benchmark.py` scores the retrieval strategies against a hand-labelled
golden set (34 queries over a fixed 25-document corpus, each query tagged
`keyword`, `paraphrase` or `mixed`). It needs no API keys, web access or databases:
embeddings run locally, and BM25 and hybrid fusion call the production functions.

```bash
python -m eval.retrieval_benchmark --strategies bm25 vector hybrid hybrid_rerank
python -m eval.retrieval_benchmark --baseline eval/results/baseline.json  # exits 1 on regression
```

Current baseline (`eval/results/baseline.json`, run `5f4ee3b0139b`):

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR | MRR on paraphrase queries |
|----------|---------:|---------:|---------:|----:|--------------------------:|
| BM25 only | 0.733 | 0.902 | 0.946 | 0.895 | 0.780 |
| Vector only | 0.723 | 0.863 | 0.926 | 0.890 | 0.804 |
| Hybrid (0.4 BM25 + 0.6 vector) | 0.821 | 0.946 | 0.976 | 0.966 | 0.917 |
| Hybrid + cross-encoder rerank | 0.851 | 0.936 | 0.976 | 0.985 | 0.964 |

Every run records a hash of everything that can change its scores (corpus, golden set,
models, top-k, chunking, fusion weight). The baseline check only compares runs with the
same hash, so a score change always means a code change, not different inputs.

Graph strategies are not covered yet because they need an LLM to build the graph.
`eval/run_benchmark.py` is a separate end-to-end smoke test: it runs full investigations
(needs API keys and web access) and checks that reports mention the expected themes.

## Not built yet

- Production async message queue (Redis/Celery), authentication, and rate limits
- Larger, independently labelled benchmark corpus; graph-strategy and answer-quality evaluation
