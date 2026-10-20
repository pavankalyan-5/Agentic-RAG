import pytest

from copilot.evals.dataset import EvalCase
from copilot.evals.retrieval_eval import (
    STRATEGIES, document_ranking, evaluate_retrieval, render_markdown, retrieval_metrics,
)
from fakes import FakeEmbedder, FakeReranker


def test_hit_rate_and_mrr():
    m = retrieval_metrics([["a", "b", "c"], ["x", "y", "z"]], [{"b"}, {"q"}], k=2)
    assert m["hit_rate@2"] == 0.5          # case 1 hit at rank 2, case 2 missed
    assert m["mrr@2"] == pytest.approx(0.25)  # (1/2 + 0) / 2
    assert m["n"] == 2


def test_rank_beyond_k_is_a_miss():
    m = retrieval_metrics([["a", "b", "c"]], [{"c"}], k=2)
    assert m["hit_rate@2"] == 0 and m["mrr@2"] == 0


def test_any_expected_source_counts_for_single_topic_cases():
    # A bug with many competing PRs lists the whole cluster: finding any of them is a hit.
    m = retrieval_metrics([["x", "pr:2", "pr:1"]], [{"pr:1", "pr:2", "pr:3"}], k=5)
    assert m["hit_rate@5"] == 1 and m["mrr@5"] == pytest.approx(1 / 2) and m["recall@5"] == 1


def test_multi_part_recall_needs_all_sources():
    ranked, expected = [["a", "c", "b"]], [{"a", "b"}]
    assert retrieval_metrics(ranked, expected, k=2, require_all=[True])["recall@2"] == 0.5
    assert retrieval_metrics(ranked, expected, k=3, require_all=[True])["recall@3"] == 1.0


def test_empty_input():
    assert retrieval_metrics([], [], k=5) == {"n": 0, "hit_rate@5": 0.0, "mrr@5": 0.0, "recall@5": 0.0}


def test_document_ranking_dedupes_chunks_preserving_rank():
    assert document_ranking(["doc:a", "doc:b", "doc:a", "pr:1", "doc:b"], k=3) == ["doc:a", "doc:b", "pr:1"]


def golden(seeded) -> list[EvalCase]:
    return [
        EvalCase(id="h1", category="history", question="ERR_TIMEOUT raised by the client",
                 expected_sources=["issue:11"], reference_answer="r"),
        EvalCase(id="d1", category="docs", question="JWT tokens for login with OAuth2",
                 expected_sources=["doc:docs/auth.md"], reference_answer="r"),
        EvalCase(id="m1", category="multi_part", question="background tasks after upgrade and their fix",
                 expected_sources=["issue:10", "pr:12"], reference_answer="r"),
        EvalCase(id="u1", category="unanswerable", question="MongoDB?"),  # no sources: skipped
    ]


def test_evaluate_retrieval_compares_all_strategies(seeded_db):
    results = evaluate_retrieval(seeded_db, FakeEmbedder(), FakeReranker(), golden(seeded_db), repo="o/r", k=5)
    assert set(results) == set(STRATEGIES) == {"dense", "sparse", "hybrid", "hybrid+rerank"}
    for strategy, r in results.items():
        assert r["metrics"]["n"] == 3, strategy  # unanswerable case skipped
        for value in (r["metrics"]["hit_rate@5"], r["metrics"]["mrr@5"], r["metrics"]["recall@5"]):
            assert 0 <= value <= 1
    assert results["sparse"]["per_case"]["h1"]["rank"] == 1  # exact identifier: BM25 nails it


def test_report_shows_strategies_and_misses(seeded_db):
    results = evaluate_retrieval(seeded_db, FakeEmbedder(), FakeReranker(), golden(seeded_db), repo="o/r", k=1)
    md = render_markdown(results, k=1)
    for strategy in STRATEGIES:
        assert f"| {strategy} |" in md
    assert "Misses" in md
