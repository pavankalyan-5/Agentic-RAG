import pytest
from pydantic import ValidationError

from copilot.retrieval.agentic import SubQuery, retrieve_for_subqueries
from fakes import FakeEmbedder, FakeReranker


def run(db, subs, **kw):
    return retrieve_for_subqueries(db, FakeEmbedder(), FakeReranker(), subs, **kw)


def test_each_subquery_contributes_its_best_hit(seeded_db):
    hits = run(seeded_db, [SubQuery(query="ERR_TIMEOUT client"), SubQuery(query="JWT login 401")], keep=1)
    assert [h.number for h in hits] == [11, 13]


def test_hits_are_reranked_against_their_own_subquery(seeded_db):
    # "background tasks upgrade" alone would never surface the JWT issue; its own sub-query must.
    hits = run(seeded_db, [SubQuery(query="background tasks upgrade"), SubQuery(query="JWT login 401")], keep=2)
    assert 13 in [h.number for h in hits]


def test_filters_passed_through(seeded_db):
    hits = run(seeded_db, [SubQuery(query="background tasks", doc_types=["pr"])], keep=4)
    assert hits and all(h.doc_type == "pr" for h in hits)


def test_state_filter_passed_through(seeded_db):
    hits = run(seeded_db, [SubQuery(query="ERR_TIMEOUT JWT background", doc_types=["issue"], state="open")], keep=4)
    assert hits and {h.number for h in hits} <= {10, 13}


def test_duplicates_removed(seeded_db):
    q = SubQuery(query="JWT login")
    hits = run(seeded_db, [q, q], keep=2)
    assert len({h.chunk_id for h in hits}) == len(hits) == 2


def test_min_score_drops_irrelevant(seeded_db):
    assert run(seeded_db, [SubQuery(query="zzz qqq")], keep=4, min_score=1.0) == []


def test_no_subqueries_returns_empty(seeded_db):
    assert run(seeded_db, []) == []


def test_subquery_rejects_unknown_doc_type():
    with pytest.raises(ValidationError):
        SubQuery(query="x", doc_types=["wiki"])
