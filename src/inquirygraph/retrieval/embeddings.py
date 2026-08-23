from functools import lru_cache

from fastembed import TextEmbedding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


@lru_cache
def get_embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vec.tolist() for vec in get_embedder().embed(texts)]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
