from inquirygraph.ingest.chunking import chunk_text


def test_chunk_text_splits_long_document():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 220 for c in chunks)


def test_chunk_text_returns_single_short_chunk():
    chunks = chunk_text("short text")
    assert chunks == ["short text"]
