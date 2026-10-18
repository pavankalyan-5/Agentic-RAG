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


def test_title_ignores_hash_comments_in_code_blocks(local_repo, tmp_path):
    write(local_repo, "docs/middleware.md", "Intro\n```python\n# Ensure all requests\napp = 1\n```\n# Middleware\n")
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "middleware")
    assert run(local_repo, tmp_path / "work")["doc:docs/middleware.md"].title == "Middleware"


def test_symlinks_are_not_followed(local_repo, tmp_path):
    # Review #6: a malicious repo could symlink docs/x.md -> ~/.ssh/id_rsa and leak it to OpenAI.
    secret = tmp_path / "outside-secret.md"
    secret.write_text("# Private\nTOP SECRET")
    (local_repo / "docs/leak.md").symlink_to(secret)
    (local_repo / "docs/alias.md").symlink_to(local_repo / "docs/guide.md")
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "symlinks")
    docs = run(local_repo, tmp_path / "work")
    assert "doc:docs/leak.md" not in docs and "doc:docs/alias.md" not in docs
    assert not any("TOP SECRET" in d.body for d in docs.values())


def test_relative_workdir(local_repo, tmp_path, monkeypatch):
    # The CLI passes a relative path (data/repos); clone must not nest it inside itself.
    monkeypatch.chdir(tmp_path)
    docs = list(iter_docs("o/r", Path("data/repos"), clone_url=f"file://{local_repo}"))
    assert len(docs) == 3
    assert (tmp_path / "data/repos/o__r/README.md").exists()


def test_second_run_fetches_new_commits(local_repo, tmp_path):
    work = tmp_path / "work"
    run(local_repo, work)
    write(local_repo, "docs/new.md", "# New page")
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "add page")
    assert "doc:docs/new.md" in run(local_repo, work)


def test_skips_changelog_style_files(local_repo, tmp_path):
    # They summarize PRs we already index (with full discussion), and flood search results.
    names = ["CHANGELOG.md", "docs/release-notes.md", "docs/Release_Notes.md", "CHANGES.md", "HISTORY.md"]
    for name in names:
        write(local_repo, name, "# Changes\n- fixed things")
    write(local_repo, "docs/history-of-asgi.md", "# History of ASGI")  # a real page that merely mentions history
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-qm", "changelogs")
    docs = run(local_repo, tmp_path / "work")
    assert not {f"doc:{n}" for n in names} & set(docs)
    assert "doc:docs/history-of-asgi.md" in docs
