"""The golden dataset: a fixed set of questions with known-good sources and reference answers.

Stored as JSONL (one case per line: easy to diff and review in a PR). Cases point at DOCUMENTS
(source_id like "doc:docs/lifespan.md" or "pr:3564"), never chunk ids: re-indexing renumbers chunks.
"""

import json
from pathlib import Path
from typing import Literal

import psycopg
from pydantic import BaseModel, ValidationError, model_validator

Category = Literal["docs", "history", "multi_part", "unanswerable", "adversarial"]
ANSWERABLE = ("docs", "history", "multi_part")


class EvalCase(BaseModel):
    id: str
    category: Category
    question: str
    # Document source_ids; empty for unanswerable/adversarial. Matching rule: for multi_part ALL are needed;
    # otherwise ANY one counts (e.g. a bug fixed after many competing PRs lists the whole cluster).
    expected_sources: list[str] = []
    reference_answer: str | None = None   # what a correct answer says (for the correctness judge)
    must_not: list[str] = []              # substrings the answer must never contain (injection checks)
    in_ci: bool = False                   # part of the small subset graded on every pull request
    notes: str = ""

    @model_validator(mode="after")
    def _answerable_cases_are_complete(self):
        if self.category in ANSWERABLE and not (self.expected_sources and self.reference_answer):
            raise ValueError(f"{self.id}: {self.category} cases need expected_sources and a reference_answer")
        return self


def load_golden(path: Path) -> list[EvalCase]:
    cases, seen = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            c = EvalCase.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{path}: line {number}: {exc}") from exc
        if c.id in seen:
            raise ValueError(f"{path}: line {number}: duplicate id {c.id}")
        seen.add(c.id)
        cases.append(c)
    return cases


def write_cases(path: Path, cases: list[EvalCase]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(c.model_dump_json(exclude_defaults=False) + "\n" for c in cases))


def validate_against_index(conn: psycopg.Connection, cases: list[EvalCase], repo: str) -> list[str]:
    """Cases whose expected sources no longer exist (renamed doc, re-synced repo) are reported, not
    silently scored as retrieval failures."""
    wanted = sorted({s for c in cases for s in c.expected_sources})
    rows = conn.execute(
        "select source_id from documents where repo = %s and source_id = any(%s)", (repo, wanted)
    ).fetchall()
    present = {r["source_id"] for r in rows}
    return [f"{c.id}: expected source {s} is not in the index"
            for c in cases for s in c.expected_sources if s not in present]
