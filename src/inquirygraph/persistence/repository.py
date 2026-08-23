"""Postgres-backed persistence for investigations.

Stores every submitted request and its final result so that the full history
survives server restarts. The in-memory job runner + LangGraph SQLite
checkpointer remain for live progress; Postgres is the durable source of truth
for completed/failed investigations.
"""

import json
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from inquirygraph.config.settings import settings
from inquirygraph.observability.logging import log


def _conn() -> "psycopg2.extensions.connection":
    return psycopg2.connect(settings.postgres_url, connect_timeout=10)


def init_db() -> None:
    """Create the investigations table if it does not exist."""
    conn = _conn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    id                    TEXT PRIMARY KEY,
                    query                 TEXT NOT NULL,
                    status                TEXT NOT NULL,
                    final_report          JSONB,
                    contradictions        JSONB,
                    citation_verification JSONB,
                    citations             JSONB,
                    errors                JSONB,
                    llm_call_count        INT DEFAULT 0,
                    iterations            INT DEFAULT 0,
                    created_at            TIMESTAMPTZ DEFAULT now(),
                    updated_at            TIMESTAMPTZ DEFAULT now()
                )
                """
            )
    finally:
        conn.close()


def _json(value: Any) -> Any:
    return psycopg2.extras.Json(value) if value is not None else None


def _coerce(row: dict) -> dict:
    """psycopg2 returns JSONB columns as strings; parse them back to objects."""
    for key in (
        "final_report",
        "contradictions",
        "citation_verification",
        "citations",
        "errors",
    ):
        val = row.get(key)
        if isinstance(val, str):
            try:
                row[key] = json.loads(val)
            except json.JSONDecodeError:
                row[key] = None
    return row


def save_request(investigation_id: str, query: str, status: str = "queued") -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO investigations (id, query, status, created_at, updated_at)
                VALUES (%s, %s, %s, now(), now())
                ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status, updated_at = now()
                """,
                (investigation_id, query, status),
            )
        conn.commit()
    finally:
        conn.close()


def update_result(
    investigation_id: str,
    status: str,
    *,
    final_report: Optional[dict] = None,
    contradictions: Optional[list] = None,
    citation_verification: Optional[list] = None,
    citations: Optional[list] = None,
    errors: Optional[list] = None,
    llm_call_count: int = 0,
    iterations: int = 0,
) -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE investigations
                SET status = %s,
                    final_report = %s,
                    contradictions = %s,
                    citation_verification = %s,
                    citations = %s,
                    errors = %s,
                    llm_call_count = %s,
                    iterations = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    status,
                    _json(final_report),
                    _json(contradictions),
                    _json(citation_verification),
                    _json(citations),
                    _json(errors),
                    llm_call_count,
                    iterations,
                    investigation_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_investigation(investigation_id: str) -> Optional[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query, status, final_report, contradictions,
                       citation_verification, citations, errors,
                       llm_call_count, iterations, created_at
                FROM investigations WHERE id = %s
                """,
                (investigation_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            return _coerce(dict(zip(cols, row)))
    finally:
        conn.close()


def list_investigations(limit: int = 50) -> list[dict]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query, status, llm_call_count, iterations, created_at
                FROM investigations
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for row in rows:
            if row.get("created_at") is not None:
                row["created_at"] = row["created_at"].isoformat()
        return rows
    finally:
        conn.close()
