"""The LLM nodes must see whole retrieved chunks, not the 400-char display excerpt."""

import pytest
from langchain_core.runnables import RunnableLambda

from inquirygraph.agent import nodes
from inquirygraph.api.schemas import (
    CitationVerificationResult,
    CoverageDecision,
    Evidence,
    InvestigationReport,
    QueryUnderstanding,
)
from inquirygraph.knowledge.neo4j_client import GraphClaim
from inquirygraph.retrieval.vector_store import EvidenceChunk

# A full-size chunk whose only distinctive fact sits past the display cut.
TAIL_FACT = "Azure Functions is 18% cheaper above three million requests"
CHUNK = ("Lambda pricing counts requests and duration. " * 12)[:600] + TAIL_FACT
CHUNK = (CHUNK + " filler" * 40)[:800]
assert CHUNK.index(TAIL_FACT) > nodes.DISPLAY_EXCERPT_CHARS


class FakeLLM:
    """Records every rendered prompt and returns a fixed structured result."""

    def __init__(self, result):
        self.result = result
        self.prompts: list[str] = []

    def with_structured_output(self, _schema):
        def respond(prompt_value):
            self.prompts.append(prompt_value.to_string())
            return self.result
        return RunnableLambda(respond)


class FakeRetriever:
    def retrieve(self, **_):
        chunk = EvidenceChunk("c1", CHUNK, "s1", "Pricing guide", "https://pricing.example")
        claim = GraphClaim("k1", "Claim " + "x" * 600, "https://claims.example", "cost")
        return [chunk], [claim]

    def close(self):
        pass


UNDERSTANDING = QueryUnderstanding(
    summary="s", key_entities=["Lambda", "Azure"], dimensions=["cost"], success_criteria="c"
)


@pytest.fixture
def retrieved(monkeypatch):
    monkeypatch.setattr(nodes, "Retriever", FakeRetriever)
    state = {"investigation_id": "inv", "user_query": "Which is cheaper?", "query_understanding": UNDERSTANDING}
    return {**state, **nodes.retrieve_evidence(state)}


def fake_main_llm(monkeypatch, result) -> FakeLLM:
    llm = FakeLLM(result)
    monkeypatch.setattr(nodes, "get_main_llm", lambda: llm)
    return llm


def test_display_excerpt_stays_capped(retrieved):
    chunk_citation, claim_citation = retrieved["citations"]
    assert len(chunk_citation.excerpt) == nodes.DISPLAY_EXCERPT_CHARS
    assert len(claim_citation.excerpt) == nodes.DISPLAY_EXCERPT_CHARS
    assert TAIL_FACT not in chunk_citation.excerpt
    # Citation, the public shape, gains no full-text field.
    assert set(chunk_citation.model_dump()) == {"source_title", "source_url", "excerpt"}


def test_evidence_keeps_full_text_aligned_with_citations(retrieved):
    citations, evidence = retrieved["citations"], retrieved["evidence"]
    assert [e.text for e in evidence] == [CHUNK, "Claim " + "x" * 600]
    assert [e.source_url for e in evidence] == [c.source_url for c in citations]


def test_synthesis_prompt_includes_text_past_the_excerpt(retrieved, monkeypatch):
    llm = fake_main_llm(monkeypatch, InvestigationReport(executive_summary="e", sections=[]))
    nodes.synthesize_report({**retrieved, "contradictions": []})
    assert TAIL_FACT in llm.prompts[0]


def test_coverage_prompt_includes_text_past_the_excerpt(retrieved, monkeypatch):
    llm = fake_main_llm(monkeypatch, CoverageDecision(needs_more_research=False, reason="ok"))
    nodes.check_coverage({**retrieved, "iteration": 0, "max_iterations": 2})
    assert TAIL_FACT in llm.prompts[0]


def test_verification_prompt_includes_text_past_the_excerpt(retrieved, monkeypatch):
    llm = FakeLLM(CitationVerificationResult())
    monkeypatch.setattr(nodes, "get_fast_llm", lambda: llm)
    report = InvestigationReport(executive_summary="e", sections=[])
    nodes.verify_citations({**retrieved, "final_report": report})
    assert TAIL_FACT in llm.prompts[0]


def evidence(*sizes: int) -> list[Evidence]:
    return [Evidence(source_title=f"t{i}", source_url=f"u{i}", text="x" * n) for i, n in enumerate(sizes)]


def test_budget_drops_whole_lowest_ranked_items():
    kept = nodes._within_budget(evidence(800, 800, 800, 800), max_chars=2000)
    assert [e.source_url for e in kept] == ["u0", "u1"]
    assert all(len(e.text) == 800 for e in kept)


def test_budget_always_keeps_the_top_item():
    kept = nodes._within_budget(evidence(3000, 100), max_chars=2000)
    assert [len(e.text) for e in kept] == [3000]


def test_budget_applies_to_synthesis(retrieved, monkeypatch):
    llm = fake_main_llm(monkeypatch, InvestigationReport(executive_summary="e", sections=[]))
    monkeypatch.setattr(nodes.settings, "max_evidence_chars", len(CHUNK))
    nodes.synthesize_report({**retrieved, "contradictions": []})
    assert TAIL_FACT in llm.prompts[0]
    assert "https://claims.example" not in llm.prompts[0]
