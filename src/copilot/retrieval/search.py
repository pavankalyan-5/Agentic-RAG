"""Dense (pgvector), sparse (BM25 via pg_search), and hybrid (Reciprocal Rank Fusion) search."""

from dataclasses import replace
from typing import Literal

import psycopg

from copilot.llm.gateway import Embedder
from copilot.observability.tracing import record_documents, span
from copilot.retrieval.models import Hit, SearchFilters

SearchMode = Literal["dense", "sparse", "hybrid"]

_HIT_COLUMNS = (
    "c.id as chunk_id, c.document_id, c.content, c.doc_type, d.title, d.url, d.number, d.state, d.author"
)


def dense_search(
    conn: psycopg.Connection, query_vec: list[float], k: int, filters: SearchFilters | None = None
) -> list[Hit]:
    where, params = _filter_sql(filters)
    rows = conn.execute(
        f"""
        select {_HIT_COLUMNS}, 1 - (c.embedding <=> %(vec)s::vector) as score
        from chunks c join documents d on d.id = c.document_id
        where true {where}
        order by c.embedding <=> %(vec)s::vector
        limit %(k)s
        """,
        {**params, "vec": query_vec, "k": k},
    ).fetchall()
    return [Hit(**r) for r in rows]


def sparse_search(
    conn: psycopg.Connection, query: str, k: int, filters: SearchFilters | None = None
) -> list[Hit]:
    if not query.strip():
        return []
    where, params = _filter_sql(filters)
    # paradedb.match tokenizes the text itself, so user punctuation can't break the query
    # (the raw `content @@@ '<string>'` form parses query syntax and errors on `?`, `(`, `AND:`).
    rows = conn.execute(
        f"""
        select {_HIT_COLUMNS}, paradedb.score(c.id) as score
        from chunks c join documents d on d.id = c.document_id
        where c.id @@@ paradedb.match('content', %(q)s) {where}
        order by score desc
        limit %(k)s
        """,
        {**params, "q": query, "k": k},
    ).fetchall()
    return [Hit(**r) for r in rows]


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, limit: int = 10) -> list[Hit]:
    """Reciprocal Rank Fusion: score(chunk) = sum over lists of 1 / (k + rank), rank starting at 1.

    Uses only ranks, not raw scores, so BM25 scores and cosine similarities never need normalizing.
    """
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(h.chunk_id, h)
    # sorted() is stable, so ties keep first-seen order.
    ordered = sorted(first_seen, key=lambda cid: scores[cid], reverse=True)
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered[:limit]]


def search(
    conn: psycopg.Connection,
    embedder: Embedder,
    query: str,
    mode: SearchMode = "hybrid",
    k: int = 10,
    filters: SearchFilters | None = None,
) -> list[Hit]:
    if not query.strip():
        return []
    with span("search", "RETRIEVER", input__value=query, copilot__mode=mode, copilot__k=k,
              copilot__filters=filters.__dict__ if filters else None) as s:
        hits = _search(conn, embedder, query, mode, k, filters)
        record_documents(s, hits)
        return hits


def _search(conn, embedder: Embedder, query: str, mode: SearchMode, k: int, filters: SearchFilters | None) -> list[Hit]:
    if mode == "sparse":
        return sparse_search(conn, query, k, filters)
    query_vec = embedder.embed([query])[0]
    if mode == "dense":
        return dense_search(conn, query_vec, k, filters)
    # Over-fetch from each side so fusion has enough candidates to reorder.
    return rrf_fuse(
        [dense_search(conn, query_vec, k * 3, filters), sparse_search(conn, query, k * 3, filters)],
        limit=k,
    )


def _filter_sql(filters: SearchFilters | None) -> tuple[str, dict]:
    if filters is None:
        return "", {}
    clauses, params = [], {}
    if filters.doc_types:
        clauses.append("c.doc_type = any(%(doc_types)s)")
        params["doc_types"] = list(filters.doc_types)
    if filters.state:
        clauses.append("d.state = %(state)s")
        params["state"] = filters.state
    if filters.updated_after:
        clauses.append("d.updated_at >= %(updated_after)s")
        params["updated_after"] = filters.updated_after
    return "".join(f" and {c}" for c in clauses), params
