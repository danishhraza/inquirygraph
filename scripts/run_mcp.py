"""Run InquiryGraph as a Model Context Protocol (MCP) server over stdio.

Requires the `mcp` package (`pip install -e ".[mcp]"`).

Usage:
    python scripts/run_mcp.py

Wire this up in an MCP client (e.g. Claude Desktop/Cursor) with:
    command: <repo>/.venv/Scripts/python.exe
    args: ["<repo>/scripts/run_mcp.py"]
"""

import importlib
import os
import threading
from pathlib import Path


def main() -> None:
    # MCP clients launch servers from their own working directory; settings read
    # .env and checkpoints.db relative to the repo, so start from there.
    os.chdir(Path(__file__).resolve().parent.parent)
    try:
        from inquirygraph.tools.mcp_server import build_server
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("MCP SDK not installed. Run: pip install -e \".[mcp]\"") from exc

    server = build_server()
    # The research pipeline is slow to import; load it while the client handshakes
    # so the first tool call doesn't pay for it.
    threading.Thread(target=importlib.import_module, args=("inquirygraph.agent.jobs",), daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
