"""Run InquiryGraph's tools as a Model Context Protocol (MCP) server.

Requires the `mcp` package (`pip install -e ".[mcp]"`).

Usage:
    python scripts/run_mcp.py

Wire this up in an MCP client (e.g. Cursor/Claude Desktop) with:
    command: python
    args: ["scripts/run_mcp.py"]
"""

from typing import Any

from inquirygraph.tools.mcp_adapter import TOOL_HANDLERS


def _build_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("MCP SDK not installed. Run: pip install -e \".[mcp]\"") from exc

    mcp = FastMCP("inquirygraph")

    @mcp.tool()
    def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
        """Search the web and return title/url/snippet results."""
        return TOOL_HANDLERS["web_search"](query, max_results)

    @mcp.tool()
    def fetch_url(url: str) -> str | None:
        """Fetch a URL and return its extracted readable text."""
        return TOOL_HANDLERS["fetch_url"](url)

    return mcp


def main() -> None:
    mcp = _build_mcp_server()
    mcp.run()


if __name__ == "__main__":
    main()