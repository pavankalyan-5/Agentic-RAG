"""Retrieval evaluation: did search put the right DOCUMENT near the top? Pure math, no LLM judge.

Each strategy gets the question as-is (no planner), so we measure search itself:
  dense, sparse (BM25), hybrid (RRF), hybrid+rerank (what the agent uses).
Chunk results are mapped to their documents and de-duplicated, because golden cases name documents.
"""

import psycopg

from copilot.evals.dataset import EvalCase
from copilot.llm.gateway import Embedder
from copilot.retrieval.rerank import Reranker, rerank
from copilot.retrieval.search import search

STRATEGIES = ("dense", "sparse", "hybrid", "hybrid+rerank")
CHUNK_POOL = 40  # chunks fetched per question, so k distinct documents remain after de-duplication


def document_ranking(source_ids: list[str], k: int) -> list[str]:
    """Chunk-level results -> document-level ranking (first occurrence wins), cut to k."""
    return list(dict.fromkeys(source_ids))[:k]


def retrieval_metrics(ranked: list[list[str]], expected: list[set[str]], k: int,
                      require_all: list[bool] | None = None) -> dict:
    """hit_rate@k: any expected doc in the top k. mrr@k: mean of 1/rank of the first expected doc (0 if
    missed). recall@k: share of expected docs found, for cases that need ALL their sources (multi-part);
    for single-topic cases (any source counts) it equals the hit."""
    require_all = require_all or [False] * len(ranked)
    hits = rr = recall = 0.0
    for docs, want, need_all in zip(ranked, expected, require_all):
        top = docs[:k]
        first = next((i for i, d in enumerate(top, start=1) if d in want), None)
        hits += first is not None
        rr += 1 / first if first else 0
        found = len(want & set(top))
        recall += found / len(want) if need_all else float(found > 0)
    n = len(ranked)
    return {"n": n, f"hit_rate@{k}": hits / n if n else 0.0, f"mrr@{k}": rr / n if n else 0.0,
            f"recall@{k}": recall / n if n else 0.0}


def evaluate_retrieval(conn: psycopg.Connection, embedder: Embedder, reranker: Reranker,
                       cases: list[EvalCase], *, repo: str, k: int = 5) -> dict:
    cases = [c for c in cases if c.expected_sources]
    doc_ids = {r["id"]: r["source_id"] for r in conn.execute(
        "select id, source_id from documents where repo = %s", (repo,)).fetchall()}
    results = {}
    for strategy in STRATEGIES:
        ranked, per_case = [], {}
        for c in cases:
            if strategy == "hybrid+rerank":
                pool = search(conn, embedder, c.question, mode="hybrid", k=CHUNK_POOL)
                hits = rerank(reranker, c.question, pool, keep=len(pool))  # reorder only; no cutoff
            else:
                hits = search(conn, embedder, c.question, mode=strategy, k=CHUNK_POOL)
            docs = document_ranking([doc_ids[h.document_id] for h in hits], k)
            want = set(c.expected_sources)
            rank = next((i for i, d in enumerate(docs, start=1) if d in want), None)
            per_case[c.id] = {"rank": rank, "top": docs, "category": c.category}
            ranked.append(docs)
        results[strategy] = {
            "metrics": retrieval_metrics(ranked, [set(c.expected_sources) for c in cases], k,
                                         require_all=[c.category == "multi_part" for c in cases]),
            "per_case": per_case,
        }
    return results


def render_markdown(results: dict, k: int, cases: list[EvalCase] | None = None) -> str:
    lines = [f"# Retrieval evaluation (top {k} documents)", "",
             f"| strategy | hit rate@{k} | MRR@{k} | recall@{k} | cases |", "|---|---:|---:|---:|---:|"]
    for strategy, r in results.items():
        m = r["metrics"]
        lines.append(f"| {strategy} | {m[f'hit_rate@{k}']:.2f} | {m[f'mrr@{k}']:.2f} | "
                     f"{m[f'recall@{k}']:.2f} | {m['n']} |")

    categories = sorted({pc["category"] for r in results.values() for pc in r["per_case"].values()})
    lines += ["", "## Hit rate by category", "", "| strategy | " + " | ".join(categories) + " |",
              "|---|" + "---:|" * len(categories)]
    for strategy, r in results.items():
        cells = []
        for cat in categories:
            ranks = [pc["rank"] for pc in r["per_case"].values() if pc["category"] == cat]
            cells.append(f"{sum(x is not None for x in ranks) / len(ranks):.2f} ({len(ranks)})")
        lines.append(f"| {strategy} | " + " | ".join(cells) + " |")

    questions = {c.id: c.question for c in cases or []}
    lines += ["", f"## Misses (expected document not in the top {k})", ""]
    for strategy, r in results.items():
        missed = [cid for cid, pc in r["per_case"].items() if pc["rank"] is None]
        lines.append(f"**{strategy}** ({len(missed)}): " + (", ".join(
            f"`{cid}`" + (f" ({questions[cid][:60]}…)" if cid in questions else "") for cid in missed) or "none"))
        lines.append("")
    return "\n".join(lines)
