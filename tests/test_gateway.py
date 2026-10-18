import pytest

from copilot.llm.gateway import BudgetExceeded, LLMGateway
from copilot.llm.pricing import cost_usd
from fakes import FakeOpenAI


def test_cost_usd_embedding():
    assert cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)


def test_cost_usd_unknown_model_raises():
    with pytest.raises(KeyError):
        cost_usd("no-such-model", 10)


def test_embed_records_usage_and_caches(db, settings):
    fake = FakeOpenAI()
    gw = LLMGateway(db, settings, client=fake)
    v1 = gw.embed(["hello", "world"])
    v2 = gw.embed(["hello"])
    assert fake.calls == 1  # second call fully cached
    assert v2[0] == v1[0] and len(v1[0]) == 1536
    rows = db.execute("select cached from llm_usage order by id").fetchall()
    assert [r["cached"] for r in rows] == [False, True]


def test_embed_only_sends_uncached_texts(db, settings):
    fake = FakeOpenAI()
    gw = LLMGateway(db, settings, client=fake)
    gw.embed(["a"])
    out = gw.embed(["a", "b"])
    assert fake.last_input == ["b"]
    assert len(out) == 2


def test_budget_ceiling_blocks_calls(db, settings):
    settings.budget_usd = 0.001
    db.execute(
        "insert into llm_usage (model, operation, input_tokens, cost_usd, cached)"
        " values ('x', 'embed', 0, 0.001, false)"
    )
    db.commit()
    fake = FakeOpenAI()
    with pytest.raises(BudgetExceeded):
        LLMGateway(db, settings, client=fake).embed(["new text"])
    assert fake.calls == 0
