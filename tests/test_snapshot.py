import gzip
import json
import tarfile

import psycopg
import pytest
from psycopg.types.json import Jsonb

from conftest import ADMIN_URL
from copilot.db.connection import apply_schema, connect
from copilot.evals.snapshot import SNAPSHOT_TABLES, SnapshotError, export_snapshot, import_snapshot

SECOND_DB = "copilot_test_snapshot"


@pytest.fixture
def target():
    """A second, empty database: the snapshot must restore into a fresh install (like CI)."""
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {SECOND_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {SECOND_DB}")
    conn = connect(ADMIN_URL.rsplit("/", 1)[0] + f"/{SECOND_DB}")
    apply_schema(conn)
    yield conn
    conn.close()


def counts(conn) -> dict:
    return {t: conn.execute(f"select count(*) as n from {t}").fetchone()["n"] for t in SNAPSHOT_TABLES}


@pytest.fixture
def source(seeded_db):
    seeded_db.execute("insert into llm_cache (key, model, response) values ('k1', 'm', %s)",
                      (Jsonb({"answer": "tab\there, newline\nhere, quote \" and back\\slash"}),))
    seeded_db.execute("insert into llm_usage (model, operation, input_tokens, cost_usd, cached) "
                      "values ('m', 'plan', 10, 0.0123, false)")
    seeded_db.commit()
    return seeded_db


def test_round_trip_preserves_rows_embeddings_and_tricky_text(source, target, tmp_path):
    path = tmp_path / "snap.tar.gz"
    manifest = export_snapshot(source, path)
    assert manifest["tables"]["chunks"]["rows"] == counts(source)["chunks"] > 0

    import_snapshot(target, path)
    assert counts(target) == counts(source)
    q = "select c.id, c.embedding::text as e, c.content from chunks c order by id"
    assert target.execute(q).fetchall() == source.execute(q).fetchall()
    assert target.execute("select response from llm_cache").fetchone() == \
        source.execute("select response from llm_cache").fetchone()
    assert float(target.execute("select sum(cost_usd) as s from llm_usage").fetchone()["s"]) == 0.0123


def test_imported_index_is_searchable(source, target, tmp_path):
    # BM25 + HNSW indexes must be rebuilt by the COPY, not left empty.
    from copilot.retrieval.search import search
    from fakes import FakeEmbedder

    export_snapshot(source, tmp_path / "s.tar.gz")
    import_snapshot(target, tmp_path / "s.tar.gz")
    for mode in ("sparse", "dense", "hybrid"):
        assert search(target, FakeEmbedder(), "ERR_TIMEOUT", mode=mode, k=3), mode


def test_sequences_continue_after_import(source, target, tmp_path):
    export_snapshot(source, tmp_path / "s.tar.gz")
    import_snapshot(target, tmp_path / "s.tar.gz")
    top = target.execute("select max(id) as m from documents").fetchone()["m"]
    new = target.execute("insert into documents (repo, source_id, doc_type, title, url, updated_at, content_hash) "
                         "values ('o/r', 'doc:new.md', 'doc', 't', 'u', now(), 'h') returning id").fetchone()["id"]
    assert new > top  # no primary-key collision with restored rows


def test_import_replaces_existing_rows(source, target, tmp_path):
    export_snapshot(source, tmp_path / "s.tar.gz")
    import_snapshot(target, tmp_path / "s.tar.gz")
    import_snapshot(target, tmp_path / "s.tar.gz")  # twice: replaced, not duplicated
    assert counts(target) == counts(source)


def test_import_leaves_threads_and_actions_alone(source, target, tmp_path):
    target.execute("insert into thread_metadata (thread_id, status) values ('t1', 'done')")
    target.commit()
    export_snapshot(source, tmp_path / "s.tar.gz")
    import_snapshot(target, tmp_path / "s.tar.gz")
    assert target.execute("select count(*) as n from thread_metadata").fetchone()["n"] == 1


def test_import_rejects_unknown_format_and_rolls_back(source, target, tmp_path):
    path = tmp_path / "s.tar.gz"
    export_snapshot(source, path)
    import_snapshot(target, path)
    before = counts(target)

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(path) as src, tarfile.open(bad, "w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "manifest.json":
                manifest = json.loads(data)
                manifest["format"] = 999
                data = json.dumps(manifest).encode()
            member.size = len(data)
            import io
            dst.addfile(member, io.BytesIO(data))
    with pytest.raises(SnapshotError, match="format"):
        import_snapshot(target, bad)
    assert counts(target) == before  # nothing was truncated


def test_import_rejects_row_count_mismatch_and_rolls_back(source, target, tmp_path):
    path = tmp_path / "s.tar.gz"
    export_snapshot(source, path)
    truncated = tmp_path / "t.tar.gz"
    with tarfile.open(path) as src, tarfile.open(truncated, "w:gz") as dst:
        import io
        for member in src.getmembers():
            data = src.extractfile(member).read()
            if member.name == "chunks.copy":
                data = b"".join(data.splitlines(keepends=True)[:-1])  # drop the last chunk row
            member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    with pytest.raises(SnapshotError, match="chunks"):
        import_snapshot(target, truncated)
    assert counts(target)["documents"] == 0  # the whole import rolled back


def test_snapshot_is_compressed(source, tmp_path):
    path = tmp_path / "s.tar.gz"
    export_snapshot(source, path)
    with gzip.open(path) as f:
        f.read(10)  # valid gzip
