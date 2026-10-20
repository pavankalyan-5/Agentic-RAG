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


def test_cache_hits_work_without_an_api_key(db, settings):
    # CI on fork PRs has no OpenAI secret: everything cached in the eval snapshot must still be served.
    import openai

    LLMGateway(db, settings, client=FakeOpenAI()).embed(["hello"])
    keyless = LLMGateway(db, settings.model_copy(update={"openai_api_key": ""}))  # must not raise
    assert len(keyless.embed(["hello"])[0]) == 1536
    with pytest.raises(openai.OpenAIError):
        keyless.embed(["not cached"])  # a real miss still needs the key, and says so


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


def test_gateway_is_safe_to_share_between_threads(db, settings):
    # Parallel grading shares one gateway (and one DB connection) across worker threads.
    from concurrent.futures import ThreadPoolExecutor

    gw = LLMGateway(db, settings, client=FakeOpenAI())
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda i: gw.embed([f"text {i}"]), range(40)))
    assert len(results) == 40 and all(len(r[0]) == 1536 for r in results)
    assert db.execute("select count(*) as n from llm_usage").fetchone()["n"] == 40
    assert db.execute("select count(*) as n from llm_cache").fetchone()["n"] == 40
