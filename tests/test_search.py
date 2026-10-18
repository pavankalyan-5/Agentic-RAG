from datetime import datetime, timezone

import pytest

from copilot.retrieval.models import Hit, SearchFilters
from copilot.retrieval.search import dense_search, rrf_fuse, search, sparse_search
from fakes import FakeEmbedder, hash_embed


def hit(chunk_id: int) -> Hit:
    return Hit(chunk_id=chunk_id, document_id=chunk_id, content=f"c{chunk_id}", score=0.0,
               doc_type="doc", title="t", url="u", number=None)


def test_rrf_scores():
    a, b, c = hit(1), hit(2), hit(3)
    fused = rrf_fuse([[a, b], [b, c]], k=60, limit=3)
    assert [h.chunk_id for h in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_rrf_respects_limit_and_handles_empty():
    assert rrf_fuse([[], []]) == []
    assert len(rrf_fuse([[hit(i) for i in range(20)]], limit=5)) == 5


def test_sparse_finds_exact_identifier(seeded_db):
    hits = sparse_search(seeded_db, "ERR_TIMEOUT", k=3)
    assert hits[0].number == 11 and "ERR_TIMEOUT" in hits[0].content


def test_dense_returns_hits_with_metadata(seeded_db):
    hits = dense_search(seeded_db, hash_embed("JWT login"), k=2)
    assert len(hits) == 2
    assert hits[0].url.startswith("https://github.com/o/r/") and hits[0].title


@pytest.mark.parametrize("query", ['why? (JWT) "auth" AND: #4521', "", "   ", "???", "content:foo OR"])
def test_odd_queries_do_not_error(seeded_db, query):
    for mode in ("dense", "sparse", "hybrid"):
        search(seeded_db, FakeEmbedder(), query, mode=mode, k=3)


def test_filters_restrict_doc_type_and_state(seeded_db):
    hits = search(seeded_db, FakeEmbedder(), "background tasks upgrade", k=10,
                  filters=SearchFilters(doc_types=("issue",), state="open"))
    # Dense search always returns its k nearest chunks, even unrelated ones (here #13),
    # so assert the filter held and the relevant issue ranks first — not an exact list.
    assert {h.number for h in hits} <= {10, 13}
    assert hits[0].number == 10


def test_updated_after_filter(seeded_db):
    later = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert search(seeded_db, FakeEmbedder(), "background", filters=SearchFilters(updated_after=later)) == []


class SemanticEmbedder:
    """Pretends to understand meaning: maps the query to a paraphrase before embedding."""

    def __init__(self, paraphrases: dict[str, str]):
        self.paraphrases = paraphrases

    def embed(self, texts):
        return [hash_embed(self.paraphrases.get(t, t)) for t in texts]


def test_hybrid_contains_best_of_both(seeded_db):
    # Literal match on "ERR_TIMEOUT" (sparse) vs. a "meaning" match on the config doc (dense).
    query = "ERR_TIMEOUT config"
    emb = SemanticEmbedder({query: "Settings Set the request timeout in config.toml using the timeout key."})
    sparse_top = sparse_search(seeded_db, "ERR_TIMEOUT", k=1)[0].chunk_id
    dense_top = dense_search(seeded_db, emb.embed([query])[0], k=1)[0].chunk_id
    assert sparse_top != dense_top
    hybrid_ids = [h.chunk_id for h in search(seeded_db, emb, query, mode="hybrid", k=3)]
    assert sparse_top in hybrid_ids and dense_top in hybrid_ids
