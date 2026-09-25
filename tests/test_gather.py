import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from inquirygraph.agent import nodes
from inquirygraph.api.schemas import ExtractionResult, ResearchTask
from inquirygraph.tools import web_search as web
from inquirygraph.tools.web_search import FetchedSource, SearchResult


class FakeVectorStore:
    def __init__(self):
        self.urls: list[str] = []

    def upsert_chunks(self, *, source_url, **_):
        self.urls.append(source_url)


class FakeGraph:
    def init_investigation(self, *_): pass
    def upsert_knowledge(self, **_): pass
    def close(self): pass


class FakeLLM:
    def with_structured_output(self, _schema):
        return RunnableLambda(lambda _: ExtractionResult())


def task(task_id: str) -> ResearchTask:
    return ResearchTask(id=task_id, question=task_id, search_query=f"query {task_id}")


@pytest.fixture
def store(monkeypatch):
    store = FakeVectorStore()
    monkeypatch.setattr(nodes, "VectorStore", lambda: store)
    monkeypatch.setattr(nodes, "KnowledgeGraph", FakeGraph)
    monkeypatch.setattr(nodes, "get_fast_llm", FakeLLM)
    monkeypatch.setattr(nodes, "_extract_source", lambda *_: ExtractionResult())
    monkeypatch.setattr(nodes, "fetch_url", lambda url, title: FetchedSource(url, title, url, "text " * 100))
    monkeypatch.setattr(nodes.settings, "max_sources_per_task", 2)
    monkeypatch.setattr(nodes.settings, "max_total_sources", 3)
    return store


def hits(*urls: str) -> list[SearchResult]:
    return [SearchResult(title=u, url=u, snippet="") for u in urls]


def run(tasks, indexed_urls=()):
    return nodes.gather_and_index({
        "investigation_id": "inv", "user_query": "q",
        "pending_tasks": tasks, "indexed_urls": list(indexed_urls),
    })


def test_skips_sources_indexed_by_earlier_loops(store, monkeypatch):
    monkeypatch.setattr(nodes, "web_search", lambda q: hits("https://a", "https://b", "https://c"))

    result = run([task("followup-0-0")], indexed_urls=["https://a"])

    assert store.urls == ["https://b", "https://c"]
    assert result["indexed_urls"] == ["https://b", "https://c"]


def test_total_cap_spans_research_loops(store, monkeypatch):
    monkeypatch.setattr(nodes, "web_search", lambda q: hits("https://x", "https://y"))

    result = run([task("t1"), task("t2")], indexed_urls=["https://a", "https://b"])

    assert store.urls == ["https://x"]  # cap of 3 reached after one new source
    assert result["completed_task_ids"] == ["t1"]


def test_failed_search_costs_one_task(store, monkeypatch):
    def search(query):
        if query == "query t1":
            raise httpx.ConnectError("[Errno 11002] getaddrinfo failed")
        return hits("https://ok")

    monkeypatch.setattr(nodes, "web_search", search)

    result = run([task("t1"), task("t2")])

    assert store.urls == ["https://ok"]
    assert result["completed_task_ids"] == ["t2"]
    assert "getaddrinfo failed" in result["errors"][0]


def test_tavily_retries_transient_failure(monkeypatch):
    calls = []

    def post(*_, **__):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("getaddrinfo failed")
        return httpx.Response(200, json={"results": [{"url": "https://a", "title": "A", "content": "x"}]},
                              request=httpx.Request("POST", "https://api.tavily.com/search"))

    monkeypatch.setattr(web.httpx, "post", post)
    monkeypatch.setattr(web.time, "sleep", lambda _: None)

    assert [r.url for r in web._tavily_search("q", 5)] == ["https://a"]
    assert len(calls) == 2
