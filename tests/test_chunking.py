from datetime import datetime, timezone

import pytest

from copilot.ingest.chunking import chunk_markdown, chunk_thread, count_tokens
from copilot.ingest.models import Post, SourceDoc

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def thread(number: int, title: str, body: str, posts: list[str]) -> SourceDoc:
    return SourceDoc(
        repo="o/r", source_id=f"issue:{number}", doc_type="issue", number=number, title=title,
        url=f"https://github.com/o/r/issues/{number}", author="alice", labels=[], state="open",
        created_at=NOW, updated_at=NOW, body=body,
        posts=[Post(author=f"user{i}", body=p, created_at=NOW) for i, p in enumerate(posts)],
    )


@pytest.fixture
def issue_doc():
    return thread(5, "Docs typo", "There is a typo in the README.", ["Thanks, fixed in #6."])


@pytest.fixture
def long_issue_doc():
    post = " ".join(["the background task never runs after upgrading"] * 8)
    return thread(123, "Background tasks broken", "Since 0.110 tasks do not run.", [post] * 30)


@pytest.fixture
def empty_issue_doc():
    return thread(7, "Empty", "", [])


def test_markdown_splits_on_headings():
    chunks = chunk_markdown("# A\nalpha\n## B\nbeta")
    assert [c.heading for c in chunks] == ["A", "B"]
    assert "alpha" in chunks[0].text and "beta" in chunks[1].text


def test_hash_comments_in_code_blocks_are_not_headings():
    md = "# Setup\n```python\n# install deps\nimport app\n```\nDone."
    chunks = chunk_markdown(md)
    assert [c.heading for c in chunks] == ["Setup"]
    assert "# install deps" in chunks[0].text


def test_no_chunk_exceeds_limit_even_without_newlines():
    big = "word " * 20_000
    chunks = chunk_markdown(big, max_tokens=500)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 500 for c in chunks)


def test_chunk_indexes_are_sequential():
    chunks = chunk_markdown("# A\n" + "x " * 2000 + "\n# B\ny")
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_thread_kept_whole_when_small(issue_doc):
    chunks = chunk_thread(issue_doc)
    assert len(chunks) == 1
    assert "typo in the README" in chunks[0].text and "fixed in #6" in chunks[0].text


def test_long_thread_repeats_header(long_issue_doc):
    chunks = chunk_thread(long_issue_doc, max_tokens=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("#123 Background tasks broken") for c in chunks)
    assert all(count_tokens(c.text) <= 200 for c in chunks)


def test_oversized_single_post_is_hard_split():
    doc = thread(9, "Huge", "", ["word " * 5000])
    chunks = chunk_thread(doc, max_tokens=300)
    assert len(chunks) > 1 and all(count_tokens(c.text) <= 300 for c in chunks)


def test_empty_body_thread_still_chunks_title(empty_issue_doc):
    chunks = chunk_thread(empty_issue_doc)
    assert len(chunks) == 1 and chunks[0].text.startswith("#7 Empty")


def test_empty_markdown_gives_no_chunks():
    assert chunk_markdown("   ") == []
