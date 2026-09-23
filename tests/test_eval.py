import copy

import pytest

from eval.config_hash import config_hash
from eval.metrics import aggregate, recall_at_k, reciprocal_rank
from eval.retrieval_benchmark import compare, rank_documents
from inquirygraph.retrieval.strategies import fuse_scores
from inquirygraph.retrieval.vector_store import EvidenceChunk


def _chunk(chunk_id: str, source_id: str) -> EvidenceChunk:
    return EvidenceChunk(chunk_id, "text", source_id, "title", "url")


def test_recall_at_k_counts_relevant_docs_in_top_k():
    ranked = ["d1", "d2", "d3", "d4"]
    assert recall_at_k(ranked, {"d2", "d4"}, 1) == 0.0
    assert recall_at_k(ranked, {"d2", "d4"}, 3) == 0.5
    assert recall_at_k(ranked, {"d2", "d4"}, 4) == 1.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(["d1", "d2", "d3"], {"d3", "d2"}) == 0.5
    assert reciprocal_rank(["d1"], {"d9"}) == 0.0


def test_aggregate_macro_averages():
    assert aggregate([{"mrr": 1.0}, {"mrr": 0.5}]) == {"mrr": 0.75}


def test_config_hash_ignores_key_order_but_not_values():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1, "b": 2}) != config_hash({"a": 1, "b": 3})


def test_rank_documents_keeps_best_position_per_doc():
    hits = [_chunk("d2#0", "d2"), _chunk("d1#0", "d1"), _chunk("d2#1", "d2")]
    assert rank_documents(hits) == ["d2", "d1"]


def test_fuse_scores_weights_normalised_bm25_and_vector():
    chunks = [_chunk("a", "d1"), _chunk("b", "d2")]
    fused = fuse_scores(chunks, [2.0, 1.0], {"a": 0.0, "b": 1.0}, bm25_weight=0.4)
    assert [c.chunk_id for c in fused] == ["b", "a"]
    assert fused[0].score == pytest.approx(0.4 * 0.5 + 0.6 * 1.0)
    assert fused[1].score == pytest.approx(0.4 * 1.0)


def _report(mrr: float, cfg: str = "abc") -> dict:
    return {"run_id": "r", "strategies": {"hybrid": {"config_hash": cfg, "metrics": {"mrr": mrr}}}}


def test_compare_flags_regression_on_same_config():
    assert compare(_report(0.99), _report(1.0)) is True  # within tolerance
    assert compare(_report(0.90), _report(1.0)) is False


def test_compare_skips_changed_config():
    baseline = _report(1.0)
    current = copy.deepcopy(baseline)
    current["strategies"]["hybrid"].update(config_hash="xyz", metrics={"mrr": 0.1})
    assert compare(current, baseline) is True
