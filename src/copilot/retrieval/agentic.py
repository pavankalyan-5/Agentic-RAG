"""Retrieval for a planner's sub-queries: hybrid search + rerank per sub-query, then merge."""

from typing import Literal

import psycopg
from pydantic import BaseModel, Field

from copilot.llm.gateway import Embedder
from copilot.retrieval.models import Hit, SearchFilters
from copilot.retrieval.rerank import Reranker, rerank
from copilot.retrieval.search import search


class SubQuery(BaseModel):
    """One focused search. Also part of the planner's structured output, so fields are described."""

    query: str = Field(description="A self-contained search query (no pronouns referring to earlier turns).")
    doc_types: list[Literal["doc", "issue", "pr"]] | None = Field(
        default=None, description="Restrict to these source types; null searches everything."
    )
    state: Literal["open", "closed", "merged"] | None = Field(
        default=None, description="Only issues/PRs in this state; null for any."
    )


def retrieve_for_subqueries(
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker,
    sub_queries: list[SubQuery],
    k: int = 10,
    keep: int = 4,
    min_score: float | None = None,
    max_per_document: int | None = None,
) -> list[Hit]:
    """Each sub-query is reranked against ITS OWN text: for "compare X and Y", a great chunk
    about Y would score poorly against the whole question. Results keep sub-query order, and a
    chunk found by several sub-queries appears once (at its first, highest-ranked position)."""
    merged: list[Hit] = []
    seen: set[int] = set()
    for sq in sub_queries:
        filters = SearchFilters(doc_types=tuple(sq.doc_types) if sq.doc_types else None, state=sq.state)
        candidates = search(conn, embedder, sq.query, mode="hybrid", k=k, filters=filters)
        for h in rerank(reranker, sq.query, candidates, keep=keep, min_score=min_score,
                        max_per_document=max_per_document):
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                merged.append(h)
    return merged
