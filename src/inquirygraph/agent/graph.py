import sqlite3
from collections.abc import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from inquirygraph.agent.nodes import (
    check_coverage,
    detect_contradictions,
    gather_and_index,
    plan_research,
    retrieve_evidence,
    synthesize_report,
    understand_query,
    verify_citations,
)
from inquirygraph.agent.state import InvestigationState
from inquirygraph.config.settings import settings


def route_after_coverage(state: InvestigationState) -> str:
    if state.get("needs_more_research") and state.get("pending_tasks"):
        return "gather_and_index"
    return "detect_contradictions"


def build_graph(checkpointer: SqliteSaver):
    graph = StateGraph(InvestigationState)

    graph.add_node("understand_query", understand_query)
    graph.add_node("plan_research", plan_research)
    graph.add_node("gather_and_index", gather_and_index)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("check_coverage", check_coverage)
    graph.add_node("detect_contradictions", detect_contradictions)
    graph.add_node("synthesize_report", synthesize_report)
    graph.add_node("verify_citations", verify_citations)

    graph.set_entry_point("understand_query")
    graph.add_edge("understand_query", "plan_research")
    graph.add_edge("plan_research", "gather_and_index")
    graph.add_edge("gather_and_index", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "check_coverage")
    graph.add_conditional_edges(
        "check_coverage",
        route_after_coverage,
        {
            "gather_and_index": "gather_and_index",
            "detect_contradictions": "detect_contradictions",
        },
    )
    graph.add_edge("detect_contradictions", "synthesize_report")
    graph.add_edge("synthesize_report", "verify_citations")
    graph.add_edge("verify_citations", END)

    return graph.compile(checkpointer=checkpointer)


def _compile_app():
    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


def load_state(investigation_id: str) -> dict | None:
    """Latest checkpointed state of an investigation, or None if it never ran."""
    snapshot = _compile_app().get_state({"configurable": {"thread_id": investigation_id}})
    if snapshot is None or not snapshot.values:
        return None
    return snapshot.values


def run_investigation(
    investigation_id: str,
    user_query: str,
    on_node: Callable[[str], None] | None = None,
) -> InvestigationState:
    """Run the graph to completion. `on_node` is called with each node name as it finishes."""
    app = _compile_app()

    initial_state: InvestigationState = {
        "investigation_id": investigation_id,
        "user_query": user_query,
        "query_understanding": None,
        "research_plan": None,
        "pending_tasks": [],
        "completed_task_ids": [],
        "retrieved_chunk_ids": [],
        "citations": [],
        "iteration": 0,
        "max_iterations": settings.max_research_iterations,
        "needs_more_research": False,
        "coverage_decision": None,
        "final_report": None,
        "llm_call_count": 0,
        "errors": [],
        "contradictions": [],
        "citation_verification": [],
    }

    config = {"configurable": {"thread_id": investigation_id}}
    if on_node is None:
        return app.invoke(initial_state, config=config)

    # Same execution as invoke(), but yields after every node so callers can report progress.
    for update in app.stream(initial_state, config=config, stream_mode="updates"):
        for node in update:
            if not node.startswith("__"):
                on_node(node)
    return app.get_state(config).values
