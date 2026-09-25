# InquiryGraph — How It Works

A research/investigation agent. You give it a question; it does web research,
extracts facts + entities into a knowledge graph, retrieves evidence with hybrid
search, loops to fill coverage gaps, then writes a cited report and verifies it.

Example question used below: **"Compare Amazon S3 vs Azure Blob Storage for a
small startup"**

---

## Tech used at each step

| Stage | Technology |
|-------|-----------|
| Orchestration | **LangGraph** (`StateGraph`, SQLite checkpointer) |
| API | **FastAPI** + Uvicorn |
| Main LLM (plan / coverage / synthesis) | **deepseek/deepseek-chat** via OpenRouter |
| Fast LLM (extraction / contradictions / verify) | **qwen/qwen3-30b-a3b-instruct-2507** via OpenRouter |
| Web search | **Tavily** (if `TAVILY_API_KEY` set) else **DuckDuckGo** (`ddgs`) |
| Fetch + chunk | `httpx` fetch → `chunk_text` |
| Embeddings | **BAAI/bge-small-en-v1.5** via `fastembed`, **384-dim** |
| Vector store | **Qdrant** (collection `evidence_chunks`, cosine distance) |
| Knowledge graph | **Neo4j** (Bolt `7687`) — entities, claims, relations |
| Hybrid retrieval | Qdrant vector + BM25 lexical fuse (rank_bm25) |
| Reranker | **Xenova/ms-marco-MiniLM-L-6-v2** (only if `ENABLE_RERANKING=true`) |
| Durable persistence | **PostgreSQL** (`investigations` table) — survives restarts |
| Fault-tolerant state | SQLite checkpoint (LangGraph) |

---

## Step-by-step flow (concrete, with the S3 vs Blob example)

### 1. Submit (`POST /investigations`)
- FastAPI receives the query. A UUID `investigation_id` is created.
- The request is written to **Postgres** (`save_request`, status `queued`).
- In `ASYNC` mode the work is handed to a background thread (`JobRunner`).
  The API immediately returns `status: queued` and the UI starts polling.
- *LLM calls so far: 0*

### 2. Understand the question — `understand_query` (LLM: **deepseek-chat**)
- Prompt asks for: summary, key entities, dimensions, success criteria.
- For our example it returns roughly:
  - entities: `Amazon S3`, `Azure Blob Storage`, `Small Startup`
  - dimensions: `Cost`, `Performance`, `Scalability`, `Security`, `Integration`, `Ease of Use`
- *LLM calls: 1*

### 3. Plan research — `plan_research` (LLM: **deepseek-chat**)
- Produces 3–5 search tasks (capped by `MAX_TASKS_PER_PLAN`, default 4).
- Each task = one optimized `search_query`. Example tasks:
  - `Amazon S3 vs Azure Blob Storage pricing comparison for startups 2023`
  - `S3 vs Blob Storage performance benchmarks small workloads`
- Tasks are placed in `pending_tasks`.
- *LLM calls: 2*

### 4. Gather + index — `gather_and_index` (the loop body)
For **each** task (per iteration):
1. **Search** the web (`web_search`) → up to `MAX_WEB_RESULTS` (5) hits.
   - A failed search (after one retry) is recorded in `errors` and skips that task only.
2. **Fetch** up to `MAX_SOURCES_PER_TASK` (2) URLs not already indexed by an earlier
   loop; stops once the investigation has `MAX_TOTAL_SOURCES` (8) sources.
   - If fetch fails, it falls back to the search snippet.
3. **Chunk** the text (`chunk_text`).
4. **Embed** chunks with `bge-small-en-v1.5` → upsert vectors into **Qdrant** (`upsert_chunks`).
   → *This is the "embed" step.*
5. **Extract** entities + factual claims (LLM: **qwen3-30b**) → write to **Neo4j**
   (`upsert_knowledge`: `Investigation -[:HAS_CLAIM]-> Claim -[:ABOUT]-> Entity`,
   `Claim -[:FROM_SOURCE]-> Source`).
   → *This is the "make relations, store in Neo4j" step.*

So the trace you asked for is exactly:
```
question
  → plan makes q1, q2, q3, q4
  → search q1 → fetch → chunk → embed → Qdrant
               → LLM extracts claims/entities → Neo4j relations
  → search q2 ... (same)
```

### 5. Retrieve evidence — `retrieve_evidence`
- `Retriever` runs the strategy in `RETRIEVAL_STRATEGY` (default `HYBRID`):
  vector search in **Qdrant** + BM25 lexical fuse over all chunks for the query.
  (Graph retrieval only kicks in for `GRAPH` / `GRAPH_PLUS_VECTOR` modes.)
- Builds `Citation` objects (title + url + excerpt). These feed the report.

### 6. Coverage check — `check_coverage` (LLM: **deepseek-chat**)  ← loop decision
- Given the evidence, the LLM decides: enough for a cited report, or are there gaps?
- If gaps exist AND `iteration < MAX_RESEARCH_ITERATIONS`:
  it proposes **1–2 new follow-up search queries** → these become the next
  `pending_tasks` → loop back to step 4.
  - *This is where "LLM makes q2/q3…" happens — they are follow-up queries, not the original plan.*
- Otherwise it stops. `iteration` counter is incremented each pass (so
  `MAX_RESEARCH_ITERATIONS=2` can show up to 3 check_coverage passes).
- *LLM calls: +1 per pass*

### 7. Detect contradictions — `detect_contradictions` (LLM: **qwen3-30b**)
- Reads claims from **Neo4j** (`get_claims`). If ≥2 claims, the LLM looks for
  direct conflicts and persists `CONTRADICTS` relations back to Neo4j.

### 8. Synthesize report — `synthesize_report` (LLM: **deepseek-chat**)
- Builds a structured `InvestigationReport`: `executive_summary`, `sections[]`
  (title + content + citations), `contradictions[]`, `open_questions[]`.
- This is what the UI renders.

### 9. Verify citations — `verify_citations` (LLM: **qwen3-30b**)
- Checks each citation excerpt actually supports the report.

### 10. Persist + respond
- The finished result is written to **Postgres** (`update_result`) with the full
  report, contradictions, citations, and metrics.
- `GET /investigations/{id}` now returns `status: completed` with the report,
  read from Postgres (so it survives a server restart).
- `GET /investigations` lists past investigations (history panel in the UI).

---

## Key config (`.env`)

| Var | Default | Effect |
|-----|---------|--------|
| `MAX_RESEARCH_ITERATIONS` | 2 | How many times it can loop back for more research |
| `MAX_TASKS_PER_PLAN` | 4 | Tasks from the initial plan |
| `MAX_SOURCES_PER_TASK` / `MAX_TOTAL_SOURCES` | 2 / 8 | New pages fetched per task / per investigation (across loops) |
| `RETRIEVAL_STRATEGY` | hybrid | `vector` / `hybrid` / `graph` / `graph_plus_vector` |
| `ENABLE_RERANKING` | false | Turn on the cross-encoder reranker |
| `PROCESSING_MODE` | async | `async` (background) or `sync` (blocking) |

For a fast test set `MAX_RESEARCH_ITERATIONS=1` and `MAX_TASKS_PER_PLAN=2`.

---

## Inspecting Neo4j relations in real time

Open **http://localhost:7474** (Neo4j Browser), connect with
`neo4j` / `inquirygraph`, and run:

```cypher
// Everything in the graph (investigations, sources, claims, entities, relations)
MATCH (n) RETURN n LIMIT 100

// One investigation and everything it touches
MATCH (i:Investigation)-[r]->(x)
RETURN i, r, x

// Investigation -> claims -> source
MATCH (i:Investigation)-[:HAS_CLAIM]->(c:Claim)-[:FROM_SOURCE]->(s:Source)
RETURN i.id, c.text, s.url

// Claims and the entities they are ABOUT
MATCH (c:Claim)-[:ABOUT]->(e:Entity)
RETURN c.text, collect(e.name) AS entities

// Detected contradictions (after step 7)
MATCH (a:Claim)-[r:CONTRADICTS]->(b:Claim)
RETURN a.text, r.explanation, b.text

// Counts
MATCH (i:Investigation) RETURN count(i) AS investigations
MATCH (c:Claim) RETURN count(c) AS claims
MATCH (e:Entity) RETURN count(e) AS entities

// Reset the graph
MATCH (n) DETACH DELETE n
```
