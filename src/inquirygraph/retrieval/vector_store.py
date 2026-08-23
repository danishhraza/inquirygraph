import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from inquirygraph.config.settings import settings
from inquirygraph.retrieval.embeddings import EMBED_DIM, embed_query, embed_texts


@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    source_id: str
    source_title: str
    source_url: str
    score: float = 0.0


class VectorStore:
    def __init__(self) -> None:
        # Generous timeout: first-run collection creation can take several
        # seconds while Qdrant provisions storage.
        self.client = QdrantClient(url=settings.qdrant_url, timeout=120)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        investigation_id: str,
        chunks: list[str],
        source_id: str,
        source_title: str,
        source_url: str,
    ) -> list[str]:
        if not chunks:
            return []

        ids = [str(uuid.uuid4()) for _ in chunks]
        vectors = embed_texts(chunks)
        points = [
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "investigation_id": investigation_id,
                    "text": text,
                    "source_id": source_id,
                    "source_title": source_title,
                    "source_url": source_url,
                },
            )
            for chunk_id, text, vector in zip(ids, chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(
        self,
        investigation_id: str,
        query: str,
        top_k: int = 8,
    ) -> list[EvidenceChunk]:
        vector = embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="investigation_id", match=MatchValue(value=investigation_id))]
            ),
            limit=top_k,
        )
        return [
            EvidenceChunk(
                chunk_id=str(hit.id),
                text=hit.payload["text"],
                source_id=hit.payload["source_id"],
                source_title=hit.payload["source_title"],
                source_url=hit.payload["source_url"],
                score=hit.score or 0.0,
            )
            for hit in results.points
        ]

    def get_all_chunks(self, investigation_id: str) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="investigation_id",
                            match=MatchValue(value=investigation_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                chunks.append(
                    EvidenceChunk(
                        chunk_id=str(record.id),
                        text=payload.get("text", ""),
                        source_id=payload.get("source_id", ""),
                        source_title=payload.get("source_title", ""),
                        source_url=payload.get("source_url", ""),
                    )
                )
            if offset is None:
                break
        return chunks
