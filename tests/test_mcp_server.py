import time

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client

from inquirygraph.agent import jobs
from inquirygraph.api.schemas import Citation, InvestigationReport, ReportSection
from inquirygraph.tools import mcp_server

NODES = ["understand_query", "plan_research", "gather_and_index", "retrieve_evidence",
         "check_coverage", "detect_contradictions", "synthesize_report", "verify_citations"]
QUERY = "Compare Snowflake and BigQuery for a migration"


def fake_graph(investigation_id: str, user_query: str, on_node=None) -> dict:
    for node in NODES:
        time.sleep(0.03)  # long enough for the progress poller to see each step
        if on_node:
            on_node(node)
    report = InvestigationReport(
        executive_summary="BigQuery is serverless; Snowflake bills per second.",
        sections=[ReportSection(
            title="Pricing",
            content="Different billing models.",
            citations=[Citation(source_title="Docs", source_url="https://example.com", excerpt="x")],
        )],
        open_questions=["Egress costs?"],
    )
    return {
        "final_report": report,
        "llm_call_count": 7,
        "iteration": 1,
        "contradictions": [],
        "citation_verification": [{"supported": True}, {"supported": False}],
        "errors": [],
    }


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(mcp_server.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(mcp_server, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(mcp_server, "_record_request", lambda *args: None)
    monkeypatch.setattr(jobs, "run_investigation", fake_graph)
    monkeypatch.setattr(jobs, "_persist_result", lambda *args: None)
    return mcp_server.build_server()


async def test_exposes_pipeline_and_plumbing_tools(server):
    async with Client(server) as client:
        names = {tool.name for tool in (await client.list_tools()).tools}
    assert names == {"start_investigation", "get_investigation", "run_investigation",
                     "web_search", "fetch_url"}


async def test_start_then_poll_returns_report(server):
    async with Client(server) as client:
        started = (await client.call_tool("start_investigation", {"query": QUERY})).structured_content
        assert started["status"] in ("queued", "running")

        for _ in range(50):
            result = (await client.call_tool(
                "get_investigation", {"investigation_id": started["investigation_id"]}
            )).structured_content
            if result["status"] == "completed":
                break
            await anyio.sleep(0.05)

    assert result["status"] == "completed"
    assert "## Pricing" in result["report_markdown"]
    assert "[Docs](https://example.com)" in result["report_markdown"]
    assert result["citations_unsupported"] == 1


async def test_run_investigation_reports_progress_per_step(server):
    updates = []

    async def on_progress(progress, total, message):
        updates.append((progress, total, message))

    async with Client(server) as client:
        result = await client.call_tool("run_investigation", {"query": QUERY}, progress_callback=on_progress)

    assert result.structured_content["llm_call_count"] == 7
    steps = [u[0] for u in updates]
    assert len(steps) >= 4 and steps == sorted(steps)  # a notification per step it saw
    assert updates[-1][:2] == (8, 8)
    assert updates[-1][2] == "finished verify_citations"


async def test_run_investigation_surfaces_failure(server, monkeypatch):
    def broken_graph(*args, **kwargs):
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(jobs, "run_investigation", broken_graph)
    monkeypatch.setattr(jobs, "_persist_error", lambda *args: None)
    async with Client(server) as client:
        result = await client.call_tool("run_investigation", {"query": QUERY})
    assert result.is_error and "Qdrant unreachable" in result.content[0].text


async def test_rejects_short_query_and_missing_key(server, monkeypatch):
    async with Client(server) as client:
        short = await client.call_tool("start_investigation", {"query": "hi"})
        monkeypatch.setattr(mcp_server.settings, "openrouter_api_key", "")
        no_key = await client.call_tool("start_investigation", {"query": QUERY})
    assert short.is_error and "at least 10" in short.content[0].text
    assert no_key.is_error and "OPENROUTER_API_KEY" in no_key.content[0].text


async def test_unknown_investigation_is_a_tool_error(server, monkeypatch):
    monkeypatch.setattr(mcp_server, "load_state", lambda _id: None)
    async with Client(server) as client:
        result = await client.call_tool("get_investigation", {"investigation_id": "nope"})
    assert result.is_error and "No investigation" in result.content[0].text
