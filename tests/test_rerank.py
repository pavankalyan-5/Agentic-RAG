import pytest

from copilot.retrieval.models import Hit
from copilot.retrieval.rerank import CrossEncoderReranker, rerank
from fakes import FakeReranker


def hit(chunk_id: int, content: str) -> Hit:
    return Hit(chunk_id=chunk_id, document_id=chunk_id, content=content, score=0.0,
               doc_type="doc", title="t", url="u", number=None)


def test_rerank_orders_and_truncates():
    hits = [hit(1, "unrelated text"), hit(2, "lifespan startup shutdown events"), hit(3, "startup only")]
    out = rerank(FakeReranker(), "lifespan startup shutdown", hits, keep=2)
    assert [h.chunk_id for h in out] == [2, 3]
    assert out[0].score == 3  # score replaced by the reranker's


def test_rerank_min_score_drops_irrelevant():
    assert rerank(FakeReranker(), "lifespan", [hit(1, "nothing here")], keep=4, min_score=0.5) == []


def test_rerank_caps_chunks_per_document():
    # Real eval finding: three chunks of closed PR #3355 filled 3 of 4 slots and crowded out merged #3491.
    hits = [hit(1, "streaming testclient"), hit(2, "streaming testclient again"), hit(3, "streaming")]
    hits[1].document_id = 1  # chunks 1 and 2 belong to the same document
    out = rerank(FakeReranker(), "streaming testclient", hits, keep=2, max_per_document=1)
    assert [h.chunk_id for h in out] == [1, 3]


def test_rerank_empty():
    assert rerank(FakeReranker(), "q", [], keep=4) == []


def test_real_reranker_does_not_load_model_for_empty_input():
    r = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    assert r.score("q", []) == [] and r._model is None


@pytest.mark.slow
def test_real_cross_encoder_prefers_relevant_text():
    r = CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    s = r.score(
        "how do I run code at application startup",
        ["Use the lifespan handler to run code on startup and shutdown.", "Set the request timeout in config.toml."],
    )
    assert len(s) == 2 and s[0] > s[1]
