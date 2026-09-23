"""Deterministic fingerprints for benchmark runs.

Two runs are only comparable if they used the same data and the same knobs.
Hashing both into a short id makes that check mechanical instead of a matter
of memory: identical hash -> identical inputs -> any score change is noise or
a code change; different hash -> not an apples-to-apples comparison.
"""

import hashlib
import json
from pathlib import Path


def config_hash(config: dict | list) -> str:
    """Short, key-order-independent hash of a JSON-serialisable value."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def json_file_hash(path: Path) -> str:
    """Hash of a JSON file's content, immune to whitespace and CRLF/LF differences."""
    return config_hash(json.loads(path.read_text(encoding="utf-8")))
