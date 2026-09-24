"""MCP server exposing the InquiryGraph investigation pipeline to MCP clients.

Clients such as Claude Desktop, Claude Code or Cursor get the full 8-node
research graph as tools, not just the web plumbing. An investigation takes
minutes, which is longer than many clients will wait on a single tool call,
so it is offered two ways:

- start_investigation + get_investigation: returns an id at once; the client
  polls for progress.
- run_investigation: blocks until the report is ready, sending an MCP progress
  notification as each graph step finishes.

Both run on the same background JobRunner as the REST API, so persistence,
failure handling and progress tracking live in one place.

Requires the `mcp` extra: pip install -e ".[mcp]"
"""

import uuid
from typing import Any

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from inquirygraph.agent.graph import load_state
from inquirygraph.agent.jobs import JobStatus, runner
from inquirygraph.api.schemas import InvestigationReport
from inquirygraph.config.settings import settings
from inquirygraph.observability.logging import log
from inquirygraph.tools.mcp_adapter import TOOL_HANDLERS

MIN_QUERY_LENGTH = 10  # same rule as POST /investigations
# Nodes on the straight path through the graph; each extra research loop adds three.
BASE_STEPS = 8
POLL_SECONDS = 0.5

INSTRUCTIONS = """\
InquiryGraph researches a question on the web, builds a knowledge graph of the
evidence, and writes a cited report with contradictions flagged and citations
verified. An investigation takes a few minutes. Prefer start_investigation and
then poll get_investigation every 20-30 seconds; use run_investigation only if
you can wait for one long call."""


def build_server() -> MCPServer:
    mcp = MCPServer("inquirygraph", instructions=INSTRUCTIONS)

    @mcp.tool()
    def start_investigation(query: str) -> dict[str, Any]:
        """Start a research investigation in the background and return its id immediately.

        Poll get_investigation with the returned id until status is "completed".
        """
        job = _submit(query)
        return {"investigation_id": job.investigation_id, "status": job.state}

    @mcp.tool()
    def get_investigation(investigation_id: str) -> dict[str, Any]:
        """Get the status and progress of an investigation, and its report once completed."""
        job = runner.get(investigation_id)
        if job is not None:
            return _job_view(job)
        # Not started by this process: finished earlier via the API or a previous server run.
        state = load_state(investigation_id)
        if state is None:
            raise ToolError(f"No investigation with id {investigation_id}")
        return summarize(investigation_id, state)

    @mcp.tool()
    async def run_investigation(query: str, ctx: Context) -> dict[str, Any]:
        """Run a full research investigation and wait for the cited report (several minutes).

        Sends a progress notification as each step of the research graph finishes.
        """
        job = _submit(query)
        reported = 0
        while job.state in ("queued", "running"):
            await anyio.sleep(POLL_SECONDS)
            if job.steps_completed > reported:
                reported = job.steps_completed
                await ctx.report_progress(
                    reported, max(BASE_STEPS, reported), f"finished {job.current_step}"
                )
        if job.state == "failed":
            raise ToolError(f"Investigation {job.investigation_id} failed: {job.error}")
        return _job_view(job)

    @mcp.tool()
    def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web and return title/url/snippet results."""
        return TOOL_HANDLERS["web_search"](query, max_results)

    @mcp.tool()
    def fetch_url(url: str) -> str | None:
        """Fetch a URL and return its extracted readable text."""
        return TOOL_HANDLERS["fetch_url"](url)

    @mcp.resource(
        "investigation://{investigation_id}/report",
        mime_type="text/markdown",
        description="Markdown report of a completed investigation.",
    )
    def investigation_report(investigation_id: str) -> str:
        state = load_state(investigation_id)
        if state is None or state.get("final_report") is None:
            raise ValueError(f"No completed report for investigation {investigation_id}")
        return report_to_markdown(state["final_report"])

    return mcp


def summarize(investigation_id: str, state: dict) -> dict[str, Any]:
    """Compact, client-friendly view of a finished investigation's state."""
    report = state.get("final_report")
    verification = state.get("citation_verification", [])
    return {
        "investigation_id": investigation_id,
        "status": "completed",
        "report_markdown": report_to_markdown(report) if report is not None else None,
        "llm_call_count": state.get("llm_call_count", 0),
        "research_iterations": state.get("iteration", 0),
        "contradictions_found": len(state.get("contradictions", [])),
        "citations_checked": len(verification),
        "citations_unsupported": sum(1 for v in verification if not v.get("supported", True)),
        "errors": state.get("errors", []),
    }


def report_to_markdown(report: InvestigationReport | dict) -> str:
    if isinstance(report, dict):  # checkpoints may hold the serialised form
        report = InvestigationReport.model_validate(report)
    lines = ["## Executive summary", "", report.executive_summary]
    for section in report.sections:
        lines += ["", f"## {section.title}", "", section.content]
        if section.citations:
            lines += ["", "Sources:"]
            lines += [f"- [{c.source_title}]({c.source_url})" for c in section.citations]
    for heading, items in (("Contradictions", report.contradictions), ("Open questions", report.open_questions)):
        if items:
            lines += ["", f"## {heading}", ""] + [f"- {item}" for item in items]
    return "\n".join(lines)


def _job_view(job: JobStatus) -> dict[str, Any]:
    if job.state == "completed" and job.result is not None:
        return summarize(job.investigation_id, job.result)
    view = {"investigation_id": job.investigation_id, "status": job.state}
    if job.state == "failed":
        view["error"] = job.error
    else:
        view |= {"current_step": job.current_step, "steps_completed": job.steps_completed}
    return view


def _submit(query: str) -> JobStatus:
    _validate(query)
    investigation_id = str(uuid.uuid4())
    _record_request(investigation_id, query)
    log.info("mcp_investigation_started", investigation_id=investigation_id)
    return runner.submit(investigation_id, query)


def _validate(query: str) -> None:
    if len(query.strip()) < MIN_QUERY_LENGTH:
        raise ToolError(f"Query must be at least {MIN_QUERY_LENGTH} characters.")
    if not settings.openrouter_api_key:
        raise ToolError("OPENROUTER_API_KEY is not set on the InquiryGraph server.")


def _record_request(investigation_id: str, query: str) -> None:
    """Log the request in Postgres so it shows in the API/web history; optional."""
    try:
        from inquirygraph.persistence import save_request

        save_request(investigation_id, query, status="queued")
    except Exception as exc:  # noqa: BLE001 - Postgres is optional for MCP use
        log.warning("postgres_write_failed", error=str(exc))
