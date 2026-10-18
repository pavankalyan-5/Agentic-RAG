import subprocess
from pathlib import Path

import pytest

from copilot.ingest.docs_repo import iter_docs


def git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd, check=True, capture_output=True,
    )


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def local_repo(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    write(origin, "README.md", "# Project\nhello")
    write(origin, "docs/guide.md", "Intro line\n# Guide\nsteps")
    write(origin, "docs/notes.mdx", "no heading here")
    write(origin, "node_modules/pkg/x.md", "# Vendored")
    write(origin, ".github/ISSUE_TEMPLATE/bug.md", "# Bug template")
    write(origin, "src/a.py", "print(1)")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "init")
    return origin


def run(local_repo, workdir):
    return {d.source_id: d for d in iter_docs("o/r", workdir, clone_url=f"file://{local_repo}")}


def test_iter_docs_from_local_repo(local_repo, tmp_path):
    docs = run(local_repo, tmp_path / "work")
    assert sorted(docs) == ["doc:README.md", "doc:docs/guide.md", "doc:docs/notes.mdx"]
    guide = docs["doc:docs/guide.md"]
    assert guide.doc_type == "doc" and guide.number is None and guide.posts == []
    assert guide.title == "Guide"
    assert guide.url == "https://github.com/o/r/blob/main/docs/guide.md"
    assert "steps" in guide.body
    assert docs["doc:docs/notes.mdx"].title == "notes"  # no H1 -> filename stem
    assert guide.updated_at.tzinfo is not None


def test_second_run_fetches_new_commits(local_repo, tmp_path):
    work = tmp_path / "work"
    run(local_repo, work)
    write(local_repo, "docs/new.md", "# New page")
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "add page")
    assert "doc:docs/new.md" in run(local_repo, work)


def test_skips_huge_changelog(local_repo, tmp_path):
    write(local_repo, "CHANGELOG.md", "# Changes\n" + "x" * 210_000)
    write(local_repo, "docs/CHANGELOG-small.md", "# Small changes")
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "changelogs")
    docs = run(local_repo, tmp_path / "work")
    assert "doc:CHANGELOG.md" not in docs and "doc:docs/CHANGELOG-small.md" in docs
