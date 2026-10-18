from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from copilot.ingest import pipeline
from copilot.ingest.models import Post, SourceDoc
from copilot.ingest.pipeline import ingest_documents, sync_repo
from fakes import FakeEmbedder

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def doc(path="docs/guide.md", body="# Guide\nInstall with pip.") -> SourceDoc:
    return SourceDoc(
        repo="o/r", source_id=f"doc:{path}", doc_type="doc", number=None, title="Guide",
        url=f"https://github.com/o/r/blob/main/{path}", author=None, labels=[], state=None,
        created_at=None, updated_at=NOW, body=body, posts=[],
    )


def issue(number=1, body="Background tasks do not run.", state="open") -> SourceDoc:
    return SourceDoc(
        repo="o/r", source_id=f"issue:{number}", doc_type="issue", number=number, title=f"Issue {number}",
        url=f"https://github.com/o/r/issues/{number}", author="alice", labels=["bug"], state=state,
        created_at=NOW, updated_at=NOW, body=body,
        posts=[Post(author="bob", body="Same here on 0.110.", created_at=NOW)],
    )


def count(db, table: str) -> int:
    return db.execute(f"select count(*) as n from {table}").fetchone()["n"]


def test_ingest_writes_documents_and_chunks(db):
    stats = ingest_documents(db, FakeEmbedder(), [doc(), issue()])
    assert stats.seen == 2 and stats.upserted == 2 and stats.chunks_written == count(db, "chunks") > 0
    assert count(db, "documents") == 2
    row = db.execute("select embedding, doc_type from chunks where doc_type = 'issue'").fetchone()
    assert len(row["embedding"].to_list()) == 1536


def test_reingest_unchanged_is_noop(db):
    emb = FakeEmbedder()
    docs = [doc(), issue()]
    ingest_documents(db, emb, docs)
    calls, chunks = emb.calls, count(db, "chunks")
    stats = ingest_documents(db, emb, docs)
    assert stats.skipped_unchanged == 2 and stats.upserted == 0
    assert emb.calls == calls and count(db, "chunks") == chunks


def test_changed_doc_replaces_chunks(db):
    ingest_documents(db, FakeEmbedder(), [doc(body="# Guide\nold text")])
    ingest_documents(db, FakeEmbedder(), [doc(body="# Guide\nnew text")])
    contents = [r["content"] for r in db.execute("select content from chunks")]
    assert any("new text" in c for c in contents) and not any("old text" in c for c in contents)
    assert count(db, "documents") == 1


def test_metadata_change_updates_without_reembedding(db):
    emb = FakeEmbedder()
    ingest_documents(db, emb, [issue(state="open")])
    calls = emb.calls
    stats = ingest_documents(db, emb, [issue(state="closed")])
    assert emb.calls == calls and stats.skipped_unchanged == 1
    assert db.execute("select state from documents").fetchone()["state"] == "closed"


def test_pii_never_stored(db):
    ingest_documents(db, FakeEmbedder(), [issue(body="contact jane.doe@example.com please")])
    text = " ".join(r["content"] for r in db.execute("select content from chunks"))
    assert "jane.doe@example.com" not in text and "<EMAIL_ADDRESS>" in text


class FakeGitHub:
    def __init__(self, docs, fail_after=None):
        self.docs, self.fail_after, self.since_calls = docs, fail_after, []

    def iter_threads(self, repo, since):
        self.since_calls.append(since)
        for i, d in enumerate(self.docs):
            if self.fail_after is not None and i == self.fail_after:
                raise httpx.HTTPStatusError("boom", request=httpx.Request("GET", "u"), response=httpx.Response(503))
            yield d


@pytest.fixture
def no_docs(monkeypatch):
    monkeypatch.setattr(pipeline, "iter_docs", lambda repo, workdir: iter([]))


def test_failure_midway_keeps_progress_and_does_not_advance_sync_state(db, tmp_path, no_docs):
    gh = FakeGitHub([issue(1), issue(2), issue(3)], fail_after=2)
    with pytest.raises(httpx.HTTPStatusError):
        sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    assert count(db, "documents") == 2
    assert db.execute("select * from sync_state where source = 'threads'").fetchone() is None


def test_thread_progress_is_checkpointed_so_a_failed_run_still_moves_forward(db, tmp_path, no_docs, monkeypatch):
    # Review #3: without intermediate checkpoints a big first sync restarts from scratch forever.
    monkeypatch.setattr(pipeline, "CHECKPOINT_EVERY", 2)
    threads = [replace(issue(n), updated_at=NOW + timedelta(hours=n)) for n in range(1, 6)]
    gh = FakeGitHub(threads, fail_after=3)  # issues 1-3 committed, then the fetch dies
    with pytest.raises(httpx.HTTPStatusError):
        sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    mark = db.execute("select last_synced_at from sync_state where source = 'threads'").fetchone()
    assert mark["last_synced_at"] == NOW + timedelta(hours=2)  # checkpoint after the 2nd committed item

    gh2 = FakeGitHub(threads)
    sync_repo(db, FakeEmbedder(), gh2, "o/r", 12, tmp_path)
    assert gh2.since_calls[0] == NOW + timedelta(hours=2)  # resumed, not restarted


def test_checkpoint_never_passes_run_start(db, tmp_path, no_docs, monkeypatch):
    monkeypatch.setattr(pipeline, "CHECKPOINT_EVERY", 1)
    future = datetime.now(timezone.utc) + timedelta(days=1)  # clock skew / edits during the run
    gh = FakeGitHub([replace(issue(1), updated_at=future), issue(2)], fail_after=1)
    with pytest.raises(httpx.HTTPStatusError):
        sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    mark = db.execute("select last_synced_at from sync_state where source = 'threads'").fetchone()
    assert mark["last_synced_at"] < future


def test_first_sync_uses_since_months_then_last_synced_at(db, tmp_path, no_docs):
    gh = FakeGitHub([issue(1)])
    before = datetime.now(timezone.utc)
    sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    first_since = gh.since_calls[0]
    assert before - timedelta(days=366) < first_since < before - timedelta(days=359)

    stored = db.execute("select last_synced_at from sync_state where source = 'threads'").fetchone()
    sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    assert gh.since_calls[1] == stored["last_synced_at"]
    assert stored["last_synced_at"] >= before


def test_full_sync_ignores_checkpoint(db, tmp_path, no_docs):
    # After redaction/chunking rules change, unchanged-on-GitHub threads must be re-processed.
    gh = FakeGitHub([issue(1)])
    sync_repo(db, FakeEmbedder(), gh, "o/r", 1, tmp_path)
    sync_repo(db, FakeEmbedder(), gh, "o/r", 1, tmp_path, full=True)
    assert gh.since_calls[1] < datetime.now(timezone.utc) - timedelta(days=29)


def test_deleted_doc_files_are_removed(db, tmp_path, monkeypatch):
    gh = FakeGitHub([])
    monkeypatch.setattr(pipeline, "iter_docs", lambda repo, workdir: iter([doc("a.md"), doc("b.md")]))
    sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    monkeypatch.setattr(pipeline, "iter_docs", lambda repo, workdir: iter([doc("a.md")]))
    sync_repo(db, FakeEmbedder(), gh, "o/r", 12, tmp_path)
    ids = [r["source_id"] for r in db.execute("select source_id from documents")]
    assert ids == ["doc:a.md"]
