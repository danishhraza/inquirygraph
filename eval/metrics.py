"""Ranking metrics for retrieval evaluation.

All functions take a ranked list of document ids (best first) and the set of
ids a human marked as relevant for the query (the golden set).
"""

from statistics import mean


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top k results."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / position of the first relevant document, or 0 if none is retrieved."""
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def score_query(ranked: list[str], relevant: set[str], ks: list[int]) -> dict[str, float]:
    scores = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks}
    scores["mrr"] = reciprocal_rank(ranked, relevant)
    return scores


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average every metric across queries (each query counts equally)."""
    if not per_query:
        return {}
    return {name: round(mean(q[name] for q in per_query), 4) for name in per_query[0]}
