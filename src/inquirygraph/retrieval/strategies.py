from rank_bm25 import BM25Okapi

from inquirygraph.config.settings import RetrievalStrategy, settings
from inquirygraph.knowledge.neo4j_client import GraphClaim, KnowledgeGraph
from inquirygraph.retrieval.reranker import rerank
from inquirygraph.retrieval.vector_store import EvidenceChunk, VectorStore

# Weight of the normalised BM25 score in hybrid fusion; vector gets the rest.
BM25_WEIGHT = 0.4


def bm25_scores(query: str, chunks: list[EvidenceChunk]) -> list[float]:
    """Lexical relevance of every chunk to the query."""
    tokenized = [chunk.text.lower().split() for chunk in chunks]
    return list(BM25Okapi(tokenized).get_scores(query.lower().split()))


def fuse_scores(
    chunks: list[EvidenceChunk],
    lexical_scores: list[float],
    vector_score_map: dict[str, float],
    bm25_weight: float = BM25_WEIGHT,
) -> list[EvidenceChunk]:
    """Weighted-sum fusion of max-normalised BM25 and cosine scores, best first.

    Pure function so the offline benchmark (eval/) scores exactly this logic.
    """
    max_bm25 = max(lexical_scores) if lexical_scores else 1.0
    combined: list[tuple[float, EvidenceChunk]] = []
    for chunk, bm25_score in zip(chunks, lexical_scores):
        vec_score = vector_score_map.get(chunk.chunk_id, 0.0)
        norm_bm25 = bm25_score / max_bm25 if max_bm25 else 0.0
        fused = bm25_weight * norm_bm25 + (1 - bm25_weight) * vec_score
        combined.append((fused, EvidenceChunk(**{**chunk.__dict__, "score": fused})))
    combined.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in combined]


class Retriever:
    """Pluggable retrieval: vector, hybrid (BM25+vector), graph, or both."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()

    def close(self) -> None:
        self.knowledge_graph.close()

    def retrieve(
        self,
        investigation_id: str,
        query: str,
        entities: list[str],
        strategy: RetrievalStrategy | None = None,
        top_k: int = 8,
    ) -> tuple[list[EvidenceChunk], list[GraphClaim]]:
        mode = strategy or settings.retrieval_strategy

        vector_hits: list[EvidenceChunk] = []
        graph_hits: list[GraphClaim] = []

        if mode in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH_PLUS_VECTOR):
            if mode == RetrievalStrategy.HYBRID:
                vector_hits = self._hybrid_search(investigation_id, query, top_k)
            else:
                vector_hits = self.vector_store.search(investigation_id, query, top_k=top_k)

        if mode in (RetrievalStrategy.GRAPH, RetrievalStrategy.GRAPH_PLUS_VECTOR):
            graph_hits = self.knowledge_graph.get_claims_for_entities(investigation_id, entities)

        if vector_hits and settings.enable_reranking:
            vector_hits = rerank(query, vector_hits, top_k)
        return vector_hits, graph_hits

    def _hybrid_search(
        self,
        investigation_id: str,
        query: str,
        top_k: int,
    ) -> list[EvidenceChunk]:
        all_chunks = self.vector_store.get_all_chunks(investigation_id)
        if not all_chunks:
            return []

        lexical = bm25_scores(query, all_chunks)

        # BM25 supplies lexical coverage; only fetch a bounded vector candidate
        # set so retrieval remains fast as the investigation grows.
        vector_hits = self.vector_store.search(
            investigation_id,
            query,
            top_k=min(len(all_chunks), max(top_k * 4, 32)),
        )
        vector_score_map = {hit.chunk_id: hit.score for hit in vector_hits}

        return fuse_scores(all_chunks, lexical, vector_score_map)[:top_k]
