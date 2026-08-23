"""Model Context Protocol (MCP) adapter.

Exposes the existing tools (web_search, fetch_url, document ingestion) as MCP
tools so the same tool layer can be consumed by MCP clients (Cursor, Claude
Desktop) and by the InquiryGraph agent itself.
"""

from typing import Any

from inquirygraph.ingest.chunking import chunk_text
from inquirygraph.tools.web_search import fetch_url, web_search


def web_search_tool(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return title/url/snippet results."""
    results = web_search(query, max_results)
    return [
        {"title": r.title, "url": r.url, "snippet": r.snippet}
        for r in results
    ]


def fetch_url_tool(url: str) -> str | None:
    """Fetch and extract readable text from a URL."""
    fetched = fetch_url(url)
    return fetched.text if fetched else None


def chunk_text_tool(text: str, chunk_size: int = 800) -> list[str]:
    """Split text into overlapping chunks."""
    return chunk_text(text, chunk_size=chunk_size)


#: Flat registry of available tools, keyed by name.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for a query and return results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and extract readable text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        },
    },
]

#: Dispatch table mapping tool name → callable.
TOOL_HANDLERS = {
    "web_search": web_search_tool,
    "fetch_url": fetch_url_tool,
}


def run_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a tool by name with the given arguments."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")
    return handler(**arguments)