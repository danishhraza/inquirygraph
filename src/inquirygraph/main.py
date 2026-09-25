import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from inquirygraph.agent.graph import load_state, run_investigation
from inquirygraph.agent.jobs import runner
from inquirygraph.api.schemas import (
    Citation,
    CitationVerification,
    Contradiction,
    DocumentUploadResponse,
    InvestigationReport,
)
from inquirygraph.config.settings import ProcessingMode, settings
from inquirygraph.persistence import (
    get_investigation as pg_get,
    init_db,
    list_investigations,
    save_request,
    update_result,
)
from inquirygraph.ingest.documents import SUPPORTED_SUFFIXES, index_document
from inquirygraph.observability.logging import log


class InvestigationRequest(BaseModel):
    query: str = Field(min_length=10, examples=[
        "Compare Databricks, Snowflake, and BigQuery for migrating from an on-premise data warehouse."
    ])


class InvestigationResponse(BaseModel):
    investigation_id: str
    status: str = "completed"
    llm_call_count: int
    iterations: int
    report: InvestigationReport | None
    contradictions: list[Contradiction]
    citation_verification: list[CitationVerification]
    citations: list[Citation] = Field(default_factory=list)
    errors: list[str]


class InvestigationSummary(BaseModel):
    id: str
    query: str
    status: str
    llm_call_count: int = 0
    iterations: int = 0
    created_at: str | None = None


app = FastAPI(
    title="InquiryGraph",
    description="AI Research & Investigation Agent — LangGraph + GraphRAG + hybrid retrieval",
    version="0.1.0",
)


@app.on_event("startup")
def _startup() -> None:
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - depends on runtime
        log.warning("postgres_init_failed", error=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/investigations/{investigation_id}", response_model=InvestigationResponse)
def get_investigation(investigation_id: str):
    # Check the in-memory job runner first (async mode).
    job = runner.get(investigation_id)
    if job is not None and job.state in ("queued", "running"):
        # LangGraph checkpoints contain the latest durable metrics even while
        # the background worker is still running.
        state = load_state(investigation_id) or {}
        return InvestigationResponse(
            investigation_id=investigation_id,
            status=job.state,
            llm_call_count=state.get("llm_call_count", 0),
            iterations=state.get("iteration", 0),
            report=state.get("final_report"),
            contradictions=state.get("contradictions", []),
            citation_verification=state.get("citation_verification", []),
            citations=state.get("citations", []),
            errors=state.get("errors", []),
        )

    # Postgres is the durable source of truth for finished investigations.
    rec = _safe_pg_get(investigation_id)
    if rec is not None:
        if rec.get("status") == "failed":
            return InvestigationResponse(
                investigation_id=investigation_id,
                status="failed",
                llm_call_count=rec.get("llm_call_count", 0),
                iterations=rec.get("iterations", 0),
                report=rec.get("final_report"),
                contradictions=[],
                citation_verification=[],
                citations=rec.get("citations", []) or [],
                errors=rec.get("errors", []) or [],
            )
        return InvestigationResponse(
            investigation_id=investigation_id,
            status="completed",
            llm_call_count=rec.get("llm_call_count", 0),
            iterations=rec.get("iterations", 0),
            report=rec.get("final_report"),
            contradictions=rec.get("contradictions", []) or [],
            citation_verification=rec.get("citation_verification", []) or [],
            citations=rec.get("citations", []) or [],
            errors=rec.get("errors", []) or [],
        )

    # Fallback to the LangGraph checkpoint.
    state = load_state(investigation_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return InvestigationResponse(
        investigation_id=investigation_id,
        status="completed",
        llm_call_count=state.get("llm_call_count", 0),
        iterations=state.get("iteration", 0),
        report=state.get("final_report"),
        contradictions=state.get("contradictions", []),
        citation_verification=state.get("citation_verification", []),
        citations=state.get("citations", []),
        errors=state.get("errors", []),
    )


def _safe_pg_get(investigation_id: str):
    try:
        return pg_get(investigation_id)
    except Exception as exc:
        log.warning("postgres_read_failed", error=str(exc))
        return None


@app.get("/investigations", response_model=list[InvestigationSummary])
def get_investigations_list():
    try:
        return list_investigations()
    except Exception as exc:
        log.warning("postgres_list_failed", error=str(exc))
        return []


@app.post("/investigations/{investigation_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(investigation_id: str, file: UploadFile = File(...)):
    from pathlib import Path

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported")

    upload_dir = Path("uploads") / investigation_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / Path(file.filename or "document.txt").name
    destination.write_bytes(await file.read())
    count = index_document(str(destination), investigation_id)
    return DocumentUploadResponse(
        investigation_id=investigation_id,
        filename=destination.name,
        chunks_indexed=count,
    )


@app.post("/investigations", response_model=InvestigationResponse)
def create_investigation(body: InvestigationRequest):
    if not settings.openrouter_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.",
        )

    investigation_id = str(uuid.uuid4())
    log.info("investigation_started", investigation_id=investigation_id)

    if settings.processing_mode == ProcessingMode.ASYNC:
        save_request(investigation_id, body.query, status="queued")
        runner.submit(investigation_id, body.query)
        return InvestigationResponse(
            investigation_id=investigation_id,
            status="queued",
            llm_call_count=0,
            iterations=0,
            report=None,
            contradictions=[],
            citation_verification=[],
            citations=[],
            errors=[],
        )

    try:
        final_state = run_investigation(investigation_id, body.query)
    except Exception as exc:
        log.exception("investigation_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    update_result(
        investigation_id,
        "completed",
        final_report=final_state.get("final_report").model_dump()
        if final_state.get("final_report") is not None
        else None,
        contradictions=[c.model_dump() for c in final_state.get("contradictions", [])],
        citation_verification=final_state.get("citation_verification", []),
        citations=[c.model_dump() for c in final_state.get("citations", [])],
        errors=final_state.get("errors", []),
        llm_call_count=final_state.get("llm_call_count", 0),
        iterations=final_state.get("iteration", 0),
    )
    return InvestigationResponse(
        investigation_id=investigation_id,
        status="completed",
        llm_call_count=final_state.get("llm_call_count", 0),
        iterations=final_state.get("iteration", 0),
        report=final_state.get("final_report"),
        contradictions=final_state.get("contradictions", []),
        citation_verification=final_state.get("citation_verification", []),
        citations=final_state.get("citations", []),
        errors=final_state.get("errors", []),
    )


@app.get("/")
def index():
    from pathlib import Path

    frontend = Path(__file__).resolve().parent.parent.parent / "frontend" / "index.html"
    if frontend.exists():
        from fastapi.responses import FileResponse

        return FileResponse(frontend)
    return {"app": "InquiryGraph", "docs": "/docs"}
