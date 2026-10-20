import json

import pytest
from pydantic import ValidationError

from copilot.evals.dataset import EvalCase, load_golden, validate_against_index, write_cases
from copilot.evals.generate import DraftCase, draft_cases
from fakes import FakeChat


def case(**kw) -> EvalCase:
    base = {"id": "docs-01", "category": "docs", "question": "How do timeouts work?",
            "expected_sources": ["doc:docs/config.md"], "reference_answer": "Set the timeout key."}
    return EvalCase(**{**base, **kw})


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "golden.jsonl"
    write_cases(path, [case(), case(id="unans-01", category="unanswerable", expected_sources=[], reference_answer=None)])
    loaded = load_golden(path)
    assert [c.id for c in loaded] == ["docs-01", "unans-01"] and loaded[1].expected_sources == []
    assert all(json.loads(line) for line in path.read_text().splitlines())  # one JSON object per line


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "g.jsonl"
    write_cases(path, [case(), case()])
    with pytest.raises(ValueError, match="duplicate id docs-01"):
        load_golden(path)


def test_unknown_category_rejected():
    with pytest.raises(ValidationError):
        case(category="trivia")


def test_answerable_cases_need_sources_and_reference():
    with pytest.raises(ValidationError):
        case(expected_sources=[])
    with pytest.raises(ValidationError):
        case(reference_answer=None)


def test_bad_line_reports_its_line_number(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(case().model_dump_json() + "\n{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        load_golden(path)


def test_validation_flags_sources_missing_from_the_index(seeded_db):
    problems = validate_against_index(seeded_db, [case(), case(id="docs-02", expected_sources=["doc:docs/gone.md"])],
                                      repo="o/r")
    assert problems == ["docs-02: expected source doc:docs/gone.md is not in the index"]


def drafter(messages) -> DraftCase:
    text = messages[-1]["content"]
    title = text.split("Title: ", 1)[1].split("\n", 1)[0]
    return DraftCase(question=f"How does {title} work in practice?", reference_answer=f"Answer about {title}.",
                     answerable=True)


def test_drafts_are_anchored_to_their_documents(seeded_db):
    chat = FakeChat({"eval_draft": [drafter] * 20})
    cases = draft_cases(seeded_db, chat, repo="o/r", per_category={"docs": 2, "history": 2, "multi_part": 1}, seed=7)
    by_cat = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    assert {k: len(v) for k, v in by_cat.items()} == {"docs": 2, "history": 2, "multi_part": 1}
    assert all(c.expected_sources[0].startswith("doc:") for c in by_cat["docs"])
    assert all(c.expected_sources[0].split(":")[0] in ("issue", "pr") for c in by_cat["history"])
    assert len(by_cat["multi_part"][0].expected_sources) == 2
    assert validate_against_index(seeded_db, cases, repo="o/r") == []


def test_drafting_is_deterministic_for_a_seed(seeded_db):
    def ids(seed):
        chat = FakeChat({"eval_draft": [drafter] * 20})
        return [c.expected_sources for c in draft_cases(seeded_db, chat, repo="o/r", per_category={"docs": 2}, seed=seed)]
    assert ids(1) == ids(1)


def test_unanswerable_drafts_are_dropped(seeded_db):
    chat = FakeChat({"eval_draft": [DraftCase(question="q?", reference_answer="", answerable=False)] * 5})
    assert draft_cases(seeded_db, chat, repo="o/r", per_category={"docs": 2}, seed=1) == []


def test_draft_prompt_asks_for_different_wording(seeded_db):
    chat = FakeChat({"eval_draft": [drafter] * 5})
    draft_cases(seeded_db, chat, repo="o/r", per_category={"docs": 1}, seed=1)
    system = chat.calls_for("eval_draft")[0]["messages"][0]["content"]
    assert "own words" in system and "headings" in system


def test_seed_cases_file_is_valid():
    from pathlib import Path
    seeds = load_golden(Path(__file__).parents[1] / "evals" / "seed_cases.jsonl")
    cats = {c.category for c in seeds}
    assert cats == {"unanswerable", "adversarial"} and len(seeds) >= 10
    adversarial = [c for c in seeds if c.category == "adversarial"]
    # Most are checked by forbidden substrings; some structurally (action targets, faithfulness) - those
    # would false-fail on a refusal that quotes the request, so they explain their check in notes instead.
    assert sum(bool(c.must_not) for c in adversarial) >= 4 and all(c.notes for c in adversarial)


def test_thresholds_file():
    from pathlib import Path
    t = json.loads((Path(__file__).parents[1] / "evals" / "thresholds.json").read_text())
    assert t["faithfulness"] == 0.85 and t["hit_rate@5"] == 0.85


def test_multi_part_pairs_related_documents(seeded_db):
    # Random pairing produced nonsense ("test suite + GraphQL"); pair each anchor with its nearest
    # neighbour by embedding similarity instead.
    from copilot.evals.generate import nearest_document

    assert nearest_document(seeded_db, "o/r", "issue:10") == "pr:12"   # both about background tasks
    assert nearest_document(seeded_db, "o/r", "doc:docs/auth.md") == "issue:13"  # both about JWT login


def test_near_duplicate_documents_are_not_paired(seeded_db):
    # README.md and docs/index.md hold the same text: pairing them makes a fake "two-source" question.
    from copilot.evals.generate import nearest_document
    from copilot.ingest.pipeline import ingest_documents
    from conftest import _seed_docs
    from fakes import FakeEmbedder

    twin = next(d for d in _seed_docs() if d.source_id == "doc:docs/auth.md")
    ingest_documents(seeded_db, FakeEmbedder(), [twin.__class__(**{**twin.__dict__, "source_id": "doc:README.md"})])
    assert nearest_document(seeded_db, "o/r", "doc:docs/auth.md") not in ("doc:README.md", None)
