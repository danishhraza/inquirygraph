"""Optional local reranking for retrieved evidence."""

from functools import lru_cache

from inquirygraph.retrieval.vector_store import EvidenceChunk


@lru_cache
def _get_reranker():
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    except (ImportError, RuntimeError, ValueError):
        return None


def rerank(query: str, chunks: list[EvidenceChunk], top_k: int = 8) -> list[EvidenceChunk]:
    """Rank chunks by query relevance; retain vector ranking if reranker is unavailable."""
    model = _get_reranker()
    if model is None or not chunks:
        return chunks[:top_k]
    scores = list(model.rerank(query, [chunk.text for chunk in chunks]))
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [EvidenceChunk(**{**chunk.__dict__, "score": float(score)}) for score, chunk in ranked[:top_k]]