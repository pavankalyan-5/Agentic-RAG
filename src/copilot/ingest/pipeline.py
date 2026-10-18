"""Fetch -> redact -> chunk -> embed -> store, with incremental sync."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg

from copilot.ingest.chunking import Chunk, chunk_markdown, chunk_thread
from copilot.ingest.docs_repo import iter_docs
from copilot.ingest.github_api import GitHubClient
from copilot.ingest.models import SourceDoc
from copilot.ingest.redact import redact
from copilot.llm.gateway import Embedder


@dataclass
class IngestStats:
    seen: int = 0
    skipped_unchanged: int = 0
    upserted: int = 0
    chunks_written: int = 0


def ingest_documents(conn: psycopg.Connection, embedder: Embedder, docs: Iterable[SourceDoc]) -> IngestStats:
    stats = IngestStats()
    for raw in docs:
        stats.seen += 1
        try:
            _ingest_one(conn, embedder, raw, stats)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return stats


def _ingest_one(conn: psycopg.Connection, embedder: Embedder, raw: SourceDoc, stats: IngestStats) -> None:
    doc = _redacted(raw)
    content_hash = _content_hash(doc)
    existing = conn.execute(
        "select id, content_hash from documents where repo = %s and source_id = %s",
        (doc.repo, doc.source_id),
    ).fetchone()

    if existing and existing["content_hash"] == content_hash:
        # Text unchanged: refresh metadata (e.g. an issue was closed) but don't pay to re-embed.
        conn.execute(
            "update documents set state = %s, labels = %s, url = %s, author = %s, updated_at = %s where id = %s",
            (doc.state, doc.labels, doc.url, doc.author, doc.updated_at, existing["id"]),
        )
        stats.skipped_unchanged += 1
        return

    chunks = chunk_markdown(doc.body) if doc.doc_type == "doc" else chunk_thread(doc)
    # Embed BEFORE writing: the gateway commits its usage rows on this connection,
    # so all writes for this document must come after it to stay in one transaction.
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []

    doc_id = _upsert_document(conn, doc, content_hash)
    conn.execute("delete from chunks where document_id = %s", (doc_id,))
    _insert_chunks(conn, doc_id, doc.doc_type, chunks, vectors)
    stats.upserted += 1
    stats.chunks_written += len(chunks)


def sync_repo(
    conn: psycopg.Connection,
    embedder: Embedder,
    gh: GitHubClient,
    repo: str,
    since_months: int,
    workdir: Path,
) -> dict[str, IngestStats]:
    results: dict[str, IngestStats] = {}

    started_at = datetime.now(timezone.utc)
    seen_docs: set[str] = set()
    docs = (seen_docs.add(d.source_id) or d for d in iter_docs(repo, workdir))
    results["docs"] = ingest_documents(conn, embedder, docs)
    _delete_missing_docs(conn, repo, seen_docs)
    _mark_synced(conn, repo, "docs", started_at)

    # Record the start time BEFORE fetching so items updated during the run are picked up next time.
    started_at = datetime.now(timezone.utc)
    since = _last_synced(conn, repo, "threads") or started_at - timedelta(days=30 * since_months)
    results["threads"] = ingest_documents(conn, embedder, gh.iter_threads(repo, since))
    _mark_synced(conn, repo, "threads", started_at)

    return results


def _redacted(doc: SourceDoc) -> SourceDoc:
    return replace(
        doc,
        title=redact(doc.title),
        body=redact(doc.body),
        posts=[replace(p, body=redact(p.body)) for p in doc.posts],
    )


def _content_hash(doc: SourceDoc) -> str:
    h = hashlib.sha256()
    for part in [doc.title, doc.body, *(f"{p.author}\0{p.body}" for p in doc.posts)]:
        h.update(part.encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _upsert_document(conn: psycopg.Connection, doc: SourceDoc, content_hash: str) -> int:
    row = conn.execute(
        """
        insert into documents (repo, source_id, doc_type, number, title, url, author, labels, state,
                               created_at, updated_at, content_hash)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on conflict (repo, source_id) do update set
            doc_type = excluded.doc_type, number = excluded.number, title = excluded.title,
            url = excluded.url, author = excluded.author, labels = excluded.labels, state = excluded.state,
            created_at = excluded.created_at, updated_at = excluded.updated_at,
            content_hash = excluded.content_hash
        returning id
        """,
        (doc.repo, doc.source_id, doc.doc_type, doc.number, doc.title, doc.url, doc.author, doc.labels,
         doc.state, doc.created_at, doc.updated_at, content_hash),
    ).fetchone()
    return row["id"]


def _insert_chunks(
    conn: psycopg.Connection, doc_id: int, doc_type: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "insert into chunks (document_id, chunk_index, heading, content, doc_type, embedding)"
            " values (%s, %s, %s, %s, %s, %s)",
            [(doc_id, c.index, c.heading, c.text, doc_type, v) for c, v in zip(chunks, vectors)],
        )


def _delete_missing_docs(conn: psycopg.Connection, repo: str, seen: set[str]) -> None:
    conn.execute(
        "delete from documents where repo = %s and doc_type = 'doc' and not (source_id = any(%s))",
        (repo, list(seen)),
    )
    conn.commit()


def _last_synced(conn: psycopg.Connection, repo: str, source: str) -> datetime | None:
    row = conn.execute(
        "select last_synced_at from sync_state where repo = %s and source = %s", (repo, source)
    ).fetchone()
    return row["last_synced_at"] if row else None


def _mark_synced(conn: psycopg.Connection, repo: str, source: str, at: datetime) -> None:
    conn.execute(
        "insert into sync_state (repo, source, last_synced_at) values (%s, %s, %s)"
        " on conflict (repo, source) do update set last_synced_at = excluded.last_synced_at",
        (repo, source, at),
    )
    conn.commit()
