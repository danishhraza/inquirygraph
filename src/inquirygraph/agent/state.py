import operator
from typing import Annotated, TypedDict

from inquirygraph.api.schemas import (
    Citation,
    Contradiction,
    CoverageDecision,
    InvestigationReport,
    QueryUnderstanding,
    ResearchPlan,
    ResearchTask,
)


class InvestigationState(TypedDict):
    investigation_id: str
    user_query: str

    query_understanding: QueryUnderstanding | None
    research_plan: ResearchPlan | None
    pending_tasks: list[ResearchTask]
    completed_task_ids: Annotated[list[str], operator.add]
    indexed_urls: Annotated[list[str], operator.add]  # sources already embedded, across research loops

    # Replaced, not appended: each retrieval pass ranks every chunk indexed so far,
    # so the latest pass supersedes earlier ones. Appending repeated the first
    # pass's hits and pushed later rounds' sources past the synthesis cutoff.
    retrieved_chunk_ids: list[str]
    citations: list[Citation]

    iteration: int
    max_iterations: int
    needs_more_research: bool
    coverage_decision: CoverageDecision | None

    final_report: InvestigationReport | None
    llm_call_count: int
    errors: Annotated[list[str], operator.add]
    contradictions: list[Contradiction]
    citation_verification: list[dict]
