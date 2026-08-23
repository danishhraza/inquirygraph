from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from langchain_core.prompts import ChatPromptTemplate

from inquirygraph.agent.state import InvestigationState
from inquirygraph.api.schemas import (
    Citation,
    CitationVerificationResult,
    Contradiction,
    ContradictionList,
    CoverageDecision,
    InvestigationReport,
    QueryUnderstanding,
    ResearchPlan,
)
from inquirygraph.config.settings import settings
from inquirygraph.ingest.chunking import chunk_text
from inquirygraph.knowledge.neo4j_client import KnowledgeGraph
from inquirygraph.llm.provider import get_fast_llm, get_main_llm
from inquirygraph.observability.logging import log
from inquirygraph.retrieval.strategies import Retriever
from inquirygraph.retrieval.vector_store import VectorStore
from inquirygraph.tools.web_search import fetch_url, web_search


def _inc_calls(state: InvestigationState, n: int = 1) -> int:
    return state.get("llm_call_count", 0) + n


# ── Node 1: Understand the question ──────────────────────────────────────────

UNDERSTAND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You analyze research questions. Return structured JSON matching the schema.",
        ),
        ("human", "{query}"),
    ]
)


def understand_query(state: InvestigationState) -> dict:
    llm = get_main_llm().with_structured_output(QueryUnderstanding)
    chain = UNDERSTAND_PROMPT | llm
    result = chain.invoke({"query": state["user_query"]})
    log.info("understand_query", entities=result.key_entities, dimensions=result.dimensions)
    return {"query_understanding": result, "llm_call_count": _inc_calls(state)}


# ── Node 2: Plan research tasks ───────────────────────────────────────────────

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Create 3-5 focused web research tasks. Each task needs a search_query optimized for search engines.",
        ),
        (
            "human",
            "Question: {query}\nEntities: {entities}\nDimensions: {dimensions}\nSuccess criteria: {criteria}",
        ),
    ]
)


def plan_research(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    assert understanding is not None

    llm = get_main_llm().with_structured_output(ResearchPlan)
    chain = PLAN_PROMPT | llm
    plan = chain.invoke(
        {
            "query": state["user_query"],
            "entities": ", ".join(understanding.key_entities),
            "dimensions": ", ".join(understanding.dimensions),
            "criteria": understanding.success_criteria,
        }
    )
    # Keep the default run bounded while preserving the LLM-generated plan.
    plan.tasks = plan.tasks[: settings.max_tasks_per_plan]
    log.info("plan_research", task_count=len(plan.tasks))
    return {
        "research_plan": plan,
        "pending_tasks": plan.tasks,
        "llm_call_count": _inc_calls(state),
    }


# ── Node 3: Gather sources, embed chunks, build knowledge graph ───────────────

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract at most 12 named entities and factual claims from the text. "
            "Claims must be short, verifiable statements. Do not explain your answer.",
        ),
        ("human", "Source: {source_title}\n\n{text}"),
    ]
)

_EXTRACTION_TIMEOUT_SECONDS = 25


def _extract_source(extractor, source_title: str, text: str):
    """Run extraction with a hard node-level timeout."""
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(extractor.invoke, {"source_title": source_title, "text": text})
    try:
        return future.result(timeout=_EXTRACTION_TIMEOUT_SECONDS)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Extraction exceeded {_EXTRACTION_TIMEOUT_SECONDS}s") from exc
    finally:
        # Do not wait for a timed-out network request. The worker is discarded
        # after this source and the investigation continues with vector data.
        pool.shutdown(wait=False, cancel_futures=True)


def gather_and_index(state: InvestigationState) -> dict:
    """
    Pure code: web search + fetch.
    LLM (fast model): entity/claim extraction → Neo4j.
    Code: chunk + embed → Qdrant.
    """
    from inquirygraph.api.schemas import ExtractionResult

    investigation_id = state["investigation_id"]
    tasks = state.get("pending_tasks") or []
    if not tasks:
        return {"errors": ["No research tasks to execute"]}

    vector_store = VectorStore()
    graph = KnowledgeGraph()
    graph.init_investigation(investigation_id, state["user_query"])

    extractor = EXTRACT_PROMPT | get_fast_llm().with_structured_output(ExtractionResult)
    llm_calls = 0
    completed: list[str] = []
    errors: list[str] = []
    sources_processed = 0

    for task in tasks:
        log.info("research_task", task_id=task.id, query=task.search_query)
        results = web_search(task.search_query)
        log.info("search_complete", task_id=task.id, result_count=len(results))

        # Keep each investigation bounded: two sources per task is enough for
        # an MVP and avoids a single slow search dominating the whole run.
        for hit in results[: settings.max_sources_per_task]:
            if sources_processed >= settings.max_total_sources:
                break
            log.info("fetch_started", task_id=task.id, url=hit.url)
            fetched = fetch_url(hit.url, hit.title)
            log.info("fetch_complete", task_id=task.id, url=hit.url, fetched=bool(fetched))
            if not fetched:
                # Fall back to snippet if page fetch fails
                if len(hit.snippet) < 80:
                    continue
                fetched_text = hit.snippet
                source_id = hit.url
                source_title = hit.title
                source_url = hit.url
            else:
                fetched_text = fetched.text
                source_id = fetched.source_id
                source_title = fetched.title
                source_url = fetched.url

            chunks = chunk_text(fetched_text)
            if not chunks:
                continue

            vector_store.upsert_chunks(
                investigation_id=investigation_id,
                chunks=chunks,
                source_id=source_id,
                source_title=source_title,
                source_url=source_url,
            )
            # Count the source as soon as its evidence is indexed. Extraction
            # may fail, but the source cap must still be respected.
            sources_processed += 1
            log.info("embedding_complete", task_id=task.id, url=source_url, chunk_count=len(chunks))

            try:
                log.info("extraction_started", task_id=task.id, url=source_url)
                extraction = _extract_source(extractor, source_title, fetched_text[:2500])
                log.info("extraction_complete", task_id=task.id, url=source_url, claim_count=len(extraction.claims))
                llm_calls += 1
                graph.upsert_knowledge(
                    investigation_id=investigation_id,
                    entities=extraction.entities,
                    claims=[c.model_dump() for c in extraction.claims],
                    source_url=source_url,
                    source_title=source_title,
                )
            except Exception as exc:
                errors.append(f"Extraction failed for {source_url}: {exc}")

        completed.append(task.id)

    graph.close()
    return {
        "completed_task_ids": completed,
        "pending_tasks": [],
        "llm_call_count": state.get("llm_call_count", 0) + llm_calls,
        "errors": errors,
    }


# ── Node 4: Retrieve evidence (vector / hybrid / graph) ────────────────────────

def retrieve_evidence(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    assert understanding is not None

    retriever = Retriever()
    try:
        vector_hits, graph_hits = retriever.retrieve(
            investigation_id=state["investigation_id"],
            query=state["user_query"],
            entities=understanding.key_entities,
        )
    finally:
        retriever.close()

    citations: list[Citation] = []
    for hit in vector_hits:
        citations.append(
            Citation(
                source_title=hit.source_title,
                source_url=hit.source_url,
                excerpt=hit.text[:400],
            )
        )
    for claim in graph_hits:
        citations.append(
            Citation(
                source_title="Knowledge graph claim",
                source_url=claim.source_url,
                excerpt=claim.text[:400],
            )
        )

    log.info(
        "retrieve_evidence",
        vector_chunks=len(vector_hits),
        graph_claims=len(graph_hits),
    )
    return {
        "retrieved_chunk_ids": [h.chunk_id for h in vector_hits],
        "citations": citations,
    }


# ── Node 5: Decide if more research is needed (loop) ──────────────────────────

COVERAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide if evidence is sufficient for a cited research report. "
            "If gaps remain, suggest 1-2 follow-up search queries.",
        ),
        (
            "human",
            "Question: {query}\nDimensions: {dimensions}\n"
            "Evidence excerpts:\n{evidence}\nIteration: {iteration}/{max_iterations}",
        ),
    ]
)


def check_coverage(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    assert understanding is not None

    evidence_text = "\n---\n".join(c.excerpt for c in state.get("citations", [])[:12])
    if not evidence_text.strip():
        decision = CoverageDecision(
            needs_more_research=True,
            reason="No evidence retrieved yet",
            follow_up_queries=[f"{entity} overview" for entity in understanding.key_entities[:2]],
        )
    else:
        llm = get_main_llm().with_structured_output(CoverageDecision)
        chain = COVERAGE_PROMPT | llm
        decision = chain.invoke(
            {
                "query": state["user_query"],
                "dimensions": ", ".join(understanding.dimensions),
                "evidence": evidence_text,
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", settings.max_research_iterations),
            }
        )

    iteration = state.get("iteration", 0)
    can_continue = iteration < state.get("max_iterations", settings.max_research_iterations)
    needs_more = decision.needs_more_research and can_continue

    if needs_more and decision.follow_up_queries:
        from inquirygraph.api.schemas import ResearchTask

        follow_up_tasks = [
            ResearchTask(
                id=f"followup-{iteration}-{i}",
                question=q,
                search_query=q,
                priority="high",
            )
            for i, q in enumerate(decision.follow_up_queries[:2])
        ]
    else:
        follow_up_tasks = []

    log.info("check_coverage", needs_more=needs_more, reason=decision.reason)
    return {
        "coverage_decision": decision,
        "needs_more_research": needs_more,
        "pending_tasks": follow_up_tasks,
        "iteration": iteration + 1,
        "llm_call_count": _inc_calls(state) if evidence_text.strip() else state.get("llm_call_count", 0),
    }


# ── Node 6: Synthesize final structured report ────────────────────────────────

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a research report using ONLY the provided evidence. "
            "Every section must include citations. If contradictions are listed, "
            "surface them explicitly rather than smoothing them over.",
        ),
        (
            "human",
            "Question: {query}\nDimensions: {dimensions}\n\n"
            "Conflicting claims:\n{contradictions}\n\nEvidence:\n{evidence}",
        ),
    ]
)


def synthesize_report(state: InvestigationState) -> dict:
    understanding = state["query_understanding"]
    assert understanding is not None
    log.info(
        "synthesize_report_start",
        citations=len(state.get("citations", [])),
        contradictions=len(state.get("contradictions", [])),
    )

    evidence_blocks = []
    for i, citation in enumerate(state.get("citations", [])[:20], start=1):
        evidence_blocks.append(
            f"[{i}] {citation.source_title} ({citation.source_url})\n{citation.excerpt}"
        )
    evidence = "\n\n".join(evidence_blocks) or "No evidence available."

    contradiction_blocks = []
    for c in state.get("contradictions", []):
        contradiction_blocks.append(
            f"- \"{c.claim_a}\" vs \"{c.claim_b}\": {c.explanation}"
        )
    contradictions_text = "\n".join(contradiction_blocks) or "None detected."

    llm = get_main_llm().with_structured_output(InvestigationReport)
    chain = SYNTHESIS_PROMPT | llm
    report = chain.invoke(
        {
            "query": state["user_query"],
            "dimensions": ", ".join(understanding.dimensions),
            "contradictions": contradictions_text,
            "evidence": evidence,
        }
    )
    log.info("synthesize_report", sections=len(report.sections))
    return {"final_report": report, "llm_call_count": _inc_calls(state)}


CONTRADICTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are given factual claims extracted from multiple sources. Find pairs of claims "
        "that directly conflict with each other (they cannot both be true). "
        "Only return genuine contradictions. "
        "If no genuine contradictions exist, return an empty list. "
        "Do NOT return pairs that are compatible, merely rephrased, or different-but-not-conflicting. "
        "Never return a pair whose explanation says 'no contradiction' or 'not contradictory'.",
    ),
    ("human", "Claims:\n{claims}"),
])


def _looks_like_contradiction(claim_a: str, claim_b: str, explanation: str) -> bool:
    """Drop items the model returned that are not real conflicts.

    The fast model sometimes returns 'analysis' objects whose explanation
    explains there is no contradiction. Those must not be surfaced as conflicts.
    """
    text = f"{claim_a} {claim_b} {explanation}".lower()
    negators = (
        "no contradiction",
        "not contradictory",
        "not in conflict",
        "no conflict",
        "are compatible",
        "essentially identical",
        "essentially the same",
        "do not contradict",
    )
    if any(neg in text for neg in negators):
        return False
    # Near-identical claims are not contradictions.
    a, b = claim_a.strip().lower(), claim_b.strip().lower()
    if a == b or not a or not b:
        return False
    import difflib

    # Only treat as "same statement" when they are essentially identical.
    # A low threshold here wrongly drops real conflicts that differ only by
    # the key entity (e.g. "Azure is cheaper" vs "AWS is cheaper").
    if difflib.SequenceMatcher(None, a, b).ratio() > 0.97:
        return False
    return True


def _filter_contradictions(contradictions: list) -> list:
    return [c for c in contradictions if _looks_like_contradiction(c.claim_a, c.claim_b, c.explanation)]


def detect_contradictions(state: InvestigationState) -> dict:
    log.info("detect_contradictions_start", investigation_id=state["investigation_id"])
    graph = KnowledgeGraph()
    try:
        claims = graph.get_claims(state["investigation_id"])
        log.info("detect_contradictions_claims", count=len(claims))
        if len(claims) < 2:
            return {"contradictions": []}

        indexed = "\n".join(
            f"[{i}] {claim.text}" for i, claim in enumerate(claims[:20])
        )
        llm = get_fast_llm().with_structured_output(ContradictionList)
        try:
            result = (CONTRADICTION_PROMPT | llm).invoke({"claims": indexed})
        except Exception as exc:
            log.warning("contradiction_detection_failed", error=str(exc))
            return {"contradictions": []}

        # Persist contradictions into the session knowledge graph.
        genuine = _filter_contradictions(result.contradictions)
        try:
            graph.add_contradictions([c.model_dump() for c in genuine])
        except Exception as exc:
            log.warning("contradiction_persist_failed", error=str(exc))
    finally:
        graph.close()

    log.info("detect_contradictions", count=len(genuine))
    return {"contradictions": genuine, "llm_call_count": _inc_calls(state)}


VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Check whether each citation excerpt supports the report. Return one result per citation."),
    ("human", "Report:\n{report}\n\nCitations:\n{citations}"),
])


def verify_citations(state: InvestigationState) -> dict:
    report = state.get("final_report")
    citations = state.get("citations", [])
    log.info("verify_citations_start", has_report=report is not None, citations=len(citations))
    if not report or not citations:
        return {"citation_verification": []}
    llm = get_fast_llm().with_structured_output(CitationVerificationResult)
    result = (VERIFY_PROMPT | llm).invoke({
        "report": report.model_dump_json(),
        "citations": "\n".join(f"[{i}] {c.excerpt}" for i, c in enumerate(citations)),
    })
    return {
        "citation_verification": [item.model_dump() for item in result.results],
        "llm_call_count": _inc_calls(state),
    }
