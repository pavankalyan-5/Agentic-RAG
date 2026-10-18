from datetime import datetime, timezone

from copilot.db.connection import apply_schema


def test_schema_creates_tables(db):
    rows = db.execute("select tablename from pg_tables where schemaname='public'").fetchall()
    assert {"documents", "chunks", "sync_state", "llm_cache", "llm_usage"} <= {r["tablename"] for r in rows}


def test_apply_schema_is_idempotent(db):
    apply_schema(db)
    apply_schema(db)


def test_bm25_and_vector_queries_work(db):
    doc_id = db.execute(
        "insert into documents (repo, source_id, doc_type, title, url, updated_at, content_hash)"
        " values ('o/r', 'issue:1', 'issue', 't', 'u', %s, 'h') returning id",
        (datetime.now(timezone.utc),),
    ).fetchone()["id"]
    vec = [1.0] + [0.0] * 1535
    chunk_id = db.execute(
        "insert into chunks (document_id, chunk_index, content, doc_type, embedding)"
        " values (%s, 0, 'ERR_TIMEOUT raised', 'issue', %s) returning id",
        (doc_id, vec),
    ).fetchone()["id"]
    db.commit()

    sparse = db.execute(
        "select id from chunks where id @@@ paradedb.match('content', %s)", ("ERR_TIMEOUT",)
    ).fetchall()
    assert [r["id"] for r in sparse] == [chunk_id]

    dense = db.execute("select id from chunks order by embedding <=> %s::vector limit 1", (vec,)).fetchone()
    assert dense["id"] == chunk_id
