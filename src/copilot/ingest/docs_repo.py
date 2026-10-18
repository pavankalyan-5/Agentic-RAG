"""Markdown docs from a shallow git clone of the repo."""

import re
import subprocess
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from copilot.ingest.chunking import markdown_title
from copilot.ingest.models import SourceDoc

DOC_SUFFIXES = {".md", ".mdx"}
SKIP_DIRS = {"node_modules", ".github", "vendor", ".git"}
# Changelog-style files summarize PRs we already index (with full discussion) and flood results.
CHANGELOG_STEM = re.compile(r"^(changelog|changes|history|release[-_]?notes)$", re.IGNORECASE)


def iter_docs(
    repo: str, workdir: Path, clone_url: str | None = None, branch: str | None = None
) -> Iterator[SourceDoc]:
    checkout = _sync_checkout(repo, workdir, clone_url or f"https://github.com/{repo}.git", branch)
    branch = branch or _git(checkout, "rev-parse", "--abbrev-ref", "HEAD")
    updated_at = datetime.fromisoformat(_git(checkout, "log", "-1", "--format=%cI"))

    root = checkout.resolve()
    for path in sorted(checkout.rglob("*")):
        rel = path.relative_to(checkout)
        # Never follow symlinks: a cloned repo could point docs/x.md at ~/.ssh/id_rsa.
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        if not path.is_file() or path.suffix.lower() not in DOC_SUFFIXES or _skip(rel):
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        yield SourceDoc(
            repo=repo,
            source_id=f"doc:{rel.as_posix()}",
            doc_type="doc",
            number=None,
            title=markdown_title(body) or path.stem,
            url=f"https://github.com/{repo}/blob/{branch}/{rel.as_posix()}",
            author=None,
            labels=[],
            state=None,
            created_at=None,
            updated_at=updated_at,
            body=body,
            posts=[],
        )


def _sync_checkout(repo: str, workdir: Path, clone_url: str, branch: str | None) -> Path:
    # Absolute, because git runs with cwd=workdir: a relative target would nest inside itself.
    checkout = workdir.resolve() / repo.replace("/", "__")
    if (checkout / ".git").exists():
        ref = branch or _git(checkout, "rev-parse", "--abbrev-ref", "HEAD")
        _git(checkout, "fetch", "--depth", "1", "origin", ref)
        _git(checkout, "reset", "--hard", "FETCH_HEAD")
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        args = ["clone", "--depth", "1", *(["--branch", branch] if branch else []), clone_url, str(checkout)]
        _git(workdir, *args)
    return checkout


def _skip(rel: Path) -> bool:
    return bool(SKIP_DIRS.intersection(rel.parts[:-1])) or bool(CHANGELOG_STEM.match(rel.stem))


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()
