"""Launch the uvicorn server as a detached process that survives the
parent shell exiting (used for local browser testing)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    log = open(ROOT / "server.log", "ab", buffering=0)
    err = open(ROOT / "server.err", "ab", buffering=0)

    proc = subprocess.Popen(
        [
            str(VENV_PY),
            "-m",
            "uvicorn",
            "inquirygraph.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        stdout=log,
        stderr=err,
        close_fds=True,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    (ROOT / "server.pid").write_text(str(proc.pid))
    print(f"launched server pid={proc.pid}")


if __name__ == "__main__":
    main()
