"""Offline, reproducible retrieval benchmark.

Scores InquiryGraph's retrieval strategies against a hand-labelled golden set
using recall@k and MRR. It needs no API keys, no web access and no running
databases: the corpus is fixed, embeddings are computed locally, and cosine
search is done in memory (the same maths Qdrant performs), while BM25 and
hybrid fusion call the production functions directly.

    python -m eval.retrieval_benchmark
    python -m eval.retrieval_benchmark --strategies hybrid hybrid_rerank
    python -m eval.retrieval_benchmark --baseline eval/results/baseline.json
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from eval.config_hash import config_hash, json_file_hash
from eval.metrics import aggregate, score_query
from inquirygraph.ingest.chunking import chunk_text
from inquirygraph.retrieval.embeddings import EMBED_MODEL, embed_texts
from inquirygraph.retrieval.reranker import RERANK_MODEL, _get_reranker, rerank
from inquirygraph.retrieval.strategies import BM25_WEIGHT, bm25_scores, fuse_scores
from inquirygraph.retrieval.vector_store import EvidenceChunk

EVAL_DIR = Path(__file__).parent
CORPUS_PATH = EVAL_DIR / "data" / "corpus.json"
GOLDEN_PATH = EVAL_DIR / "data" / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"

STRATEGIES = ("bm25", "vector", "hybrid", "hybrid_rerank")
DEFAULT_STRATEGIES = ["bm25", "vector", "hybrid"]
TOP_K = 8  # matches Retriever.retrieve's default in production
KS = [1, 3, 5]
CHUNK_SIZE, CHUNK_OVERLAP = 800, 120  # chunk_text defaults used at ingest
REGRESSION_TOLERANCE = 0.02


class BenchmarkIndex:
    """In-memory stand-in for the Qdrant collection of one investigation."""

    def __init__(self, corpus: list[dict]) -> None:
        self.chunks = [
            EvidenceChunk(
                chunk_id=f"{doc['doc_id']}#{i}",
                text=text,
                source_id=doc["doc_id"],
                source_title=doc["title"],
                source_url=f"eval://{doc['doc_id']}",
            )
            for doc in corpus
            for i, text in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP))
        ]
        self.vectors = _normalise(np.array(embed_texts([c.text for c in self.chunks])))

    def vector_search(self, query: str, top_k: int) -> list[EvidenceChunk]:
        query_vec = _normalise(np.array(embed_texts([query])))[0]
        sims = self.vectors @ query_vec
        order = np.argsort(-sims)[:top_k]
        return [EvidenceChunk(**{**self.chunks[i].__dict__, "score": float(sims[i])}) for i in order]

    def search(self, strategy: str, query: str, top_k: int = TOP_K) -> list[EvidenceChunk]:
        if strategy == "bm25":
            lexical = bm25_scores(query, self.chunks)
            order = sorted(range(len(self.chunks)), key=lambda i: lexical[i], reverse=True)
            return [self.chunks[i] for i in order[:top_k]]
        if strategy == "vector":
            return self.vector_search(query, top_k)
        # Mirrors Retriever._hybrid_search: BM25 over everything, bounded vector candidates.
        candidates = self.vector_search(query, min(len(self.chunks), max(top_k * 4, 32)))
        hits = fuse_scores(
            self.chunks,
            bm25_scores(query, self.chunks),
            {hit.chunk_id: hit.score for hit in candidates},
        )[:top_k]
        if strategy == "hybrid_rerank":
            hits = rerank(query, hits, top_k)
        return hits


def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    names = [f"recall@{k}" for k in KS] + ["mrr"]
    return aggregate([{name: row[name] for name in names} for row in rows])


def rank_documents(hits: list[EvidenceChunk]) -> list[str]:
    """Collapse chunk hits to document ids, keeping each document's best position."""
    return list(dict.fromkeys(hit.source_id for hit in hits))


def strategy_config(strategy: str, data_hashes: dict[str, str]) -> dict:
    """Every input that can change this strategy's scores. Its hash is the run's identity."""
    config = {
        "strategy": strategy,
        "top_k": TOP_K,
        "ks": KS,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        **data_hashes,
    }
    if strategy != "bm25":
        config["embed_model"] = EMBED_MODEL
    if strategy.startswith("hybrid"):
        config["bm25_weight"] = BM25_WEIGHT
    if strategy == "hybrid_rerank":
        config["rerank_model"] = RERANK_MODEL
    return config


def run(strategies: list[str]) -> dict:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    data_hashes = {"corpus_hash": json_file_hash(CORPUS_PATH), "golden_hash": json_file_hash(GOLDEN_PATH)}

    if "hybrid_rerank" in strategies and _get_reranker() is None:
        # rerank() silently falls back to the input order; never report that as a reranker score.
        sys.exit("hybrid_rerank requested but the cross-encoder could not be loaded")

    index = BenchmarkIndex(corpus)
    results: dict[str, dict] = {}
    for strategy in strategies:
        config = strategy_config(strategy, data_hashes)
        per_query = []
        for case in golden:
            ranked = rank_documents(index.search(strategy, case["query"]))
            scores = score_query(ranked, set(case["relevant"]), KS)
            per_query.append({"id": case["id"], "type": case["type"], "ranked": ranked, **scores})
        results[strategy] = {
            "config_hash": config_hash(config),
            "config": config,
            "metrics": _aggregate_metrics(per_query),
            "by_query_type": {
                qtype: _aggregate_metrics([r for r in per_query if r["type"] == qtype])
                for qtype in sorted({r["type"] for r in per_query})
            },
            "per_query": per_query,
        }

    return {
        "run_id": config_hash({s: r["config_hash"] for s, r in results.items()}),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "num_queries": len(golden),
        "num_documents": len(corpus),
        "num_chunks": len(index.chunks),
        # Recorded for debugging, deliberately not hashed: they should not change scores.
        "environment": {
            "python": platform.python_version(),
            "fastembed": version("fastembed"),
            "rank_bm25": version("rank-bm25"),
        },
        "strategies": results,
    }


def print_table(report: dict) -> None:
    names = [f"recall@{k}" for k in KS] + ["mrr"]
    print(f"\nRun {report['run_id']}  ({report['num_queries']} queries, {report['num_documents']} docs)\n")
    print(f"{'strategy':<15}{'config':<14}" + "".join(f"{n:>11}" for n in names))
    for strategy, result in report["strategies"].items():
        row = "".join(f"{result['metrics'][n]:>11.3f}" for n in names)
        print(f"{strategy:<15}{result['config_hash']:<14}{row}")
    print("\nMRR by query type:")
    for strategy, result in report["strategies"].items():
        parts = ", ".join(f"{t}={m['mrr']:.3f}" for t, m in result["by_query_type"].items())
        print(f"  {strategy:<13} {parts}")


def compare(report: dict, baseline: dict) -> bool:
    """Print deltas against a baseline; return False if a comparable metric regressed."""
    print(f"\nComparison with baseline run {baseline['run_id']}:")
    ok = True
    for strategy, result in report["strategies"].items():
        base = baseline["strategies"].get(strategy)
        if base is None:
            print(f"  {strategy:<13} not in baseline")
            continue
        if base["config_hash"] != result["config_hash"]:
            print(f"  {strategy:<13} config changed ({base['config_hash']} -> {result['config_hash']}), not comparable")
            continue
        deltas = {m: result["metrics"][m] - base["metrics"][m] for m in result["metrics"]}
        regressed = [m for m, d in deltas.items() if d < -REGRESSION_TOLERANCE]
        ok = ok and not regressed
        status = f"REGRESSED on {', '.join(regressed)}" if regressed else "ok"
        print(f"  {strategy:<13} " + ", ".join(f"{m} {d:+.3f}" for m, d in deltas.items()) + f"  [{status}]")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=DEFAULT_STRATEGIES)
    parser.add_argument("--baseline", type=Path, help="fail if a metric drops vs this results file")
    parser.add_argument("--out", type=Path, help="results path (default: eval/results/<run_id>.json)")
    args = parser.parse_args()

    report = run(args.strategies)
    print_table(report)

    out = args.out or RESULTS_DIR / f"{report['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out}")

    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not compare(report, baseline):
            sys.exit(1)


if __name__ == "__main__":
    main()
