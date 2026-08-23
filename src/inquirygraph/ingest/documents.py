"""Local document ingestion for text, Markdown, and PDF files."""

from pathlib import Path

from inquirygraph.ingest.chunking import chunk_text
from inquirygraph.retrieval.vector_store import VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def extract_text(path: Path) -> str:
    """Extract raw text from a supported document."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    raise ValueError("Only .txt, .md, and .pdf documents are supported")


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def index_document(path: str, investigation_id: str) -> int:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Only .txt, .md, and .pdf documents are supported")

    text = extract_text(file_path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    VectorStore().upsert_chunks(
        investigation_id,
        chunks,
        source_id=str(file_path),
        source_title=file_path.name,
        source_url=f"file://{file_path.resolve()}",
    )
    return len(chunks)