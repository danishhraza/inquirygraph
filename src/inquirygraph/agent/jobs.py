"""In-memory background job runner for investigations.

Lets the API accept a request and process it in a worker thread while callers
poll `GET /investigations/{id}` for the result. Kept intentionally simple —
LangGraph's SQLite checkpointing already persists the durable state, so this
module only tracks *whether* a job is currently running.
"""

import threading
from dataclasses import dataclass

from inquirygraph.agent.graph import run_investigation
from inquirygraph.observability.logging import log


@dataclass
class JobStatus:
    investigation_id: str
    state: str = "queued"  # queued | running | completed | failed
    error: str | None = None
    result: dict | None = None
    current_step: str | None = None  # last graph node that finished
    steps_completed: int = 0


class JobRunner:
    """Thread-based executor with a shared job registry."""

    def __init__(self) -> None:
        self._executor = _ThreadPool()
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def submit(self, investigation_id: str, user_query: str) -> JobStatus:
        job = JobStatus(investigation_id=investigation_id, state="queued")
        with self._lock:
            self._jobs[investigation_id] = job
        self._executor.submit(self._run, job, user_query)
        return job

    def get(self, investigation_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(investigation_id)

    def _run(self, job: JobStatus, user_query: str) -> None:
        with self._lock:
            job.state = "running"
        log.info("job_started", investigation_id=job.investigation_id)
        try:
            result = run_investigation(
                job.investigation_id, user_query, on_node=lambda node: self._advance(job, node)
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - depends on runtime
            log.exception("job_failed", investigation_id=job.investigation_id)
            with self._lock:
                job.error = str(exc)
                job.state = "failed"
            _best_effort(_persist_error, job.investigation_id, str(exc))
            return

        with self._lock:
            job.result = result
            job.state = "completed"
        # Postgres is optional (the checkpoint already holds the result), so an
        # outage there must not turn a finished investigation into a failure.
        _best_effort(_persist_result, job.investigation_id, "completed", result)
        log.info(
            "job_completed",
            investigation_id=job.investigation_id,
            has_report=result.get("final_report") is not None,
            llm_calls=result.get("llm_call_count"),
            iterations=result.get("iteration"),
        )

    def _advance(self, job: JobStatus, node: str) -> None:
        with self._lock:
            job.current_step = node
            job.steps_completed += 1


class _ThreadPool:
    """Tiny wrapper over ThreadPoolExecutor that keeps a bounded worker count."""

    def __init__(self, max_workers: int = 4) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, *args) -> None:
        self._pool.submit(fn, *args)


def _best_effort(persist, *args) -> None:
    try:
        persist(*args)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - depends on runtime
        log.warning("postgres_write_failed", error=str(exc))


def _persist_result(investigation_id: str, status: str, result: dict) -> None:
    """Persist a finished investigation to Postgres (best-effort)."""
    from inquirygraph.persistence import update_result

    final_report = result.get("final_report")
    update_result(
        investigation_id,
        status,
        final_report=final_report.model_dump() if final_report is not None else None,
        contradictions=[c.model_dump() for c in result.get("contradictions", [])],
        citation_verification=result.get("citation_verification", []),
        citations=[c.model_dump() for c in result.get("citations", [])],
        errors=result.get("errors", []),
        llm_call_count=result.get("llm_call_count", 0),
        iterations=result.get("iteration", 0),
    )


def _persist_error(investigation_id: str, error: str) -> None:
    from inquirygraph.persistence import update_result

    update_result(investigation_id, "failed", errors=[error])


#: Module-level singleton used by the API.
runner = JobRunner()