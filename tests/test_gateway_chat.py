import pytest
from pydantic import BaseModel

from copilot.llm.gateway import BudgetExceeded, LLMGateway, LLMOutputError
from fakes import FakeOpenAI


class Echo(BaseModel):
    text: str


MSGS = [{"role": "user", "content": "x"}]


def test_chat_returns_parsed_schema_and_records_cost(db, settings):
    fake = FakeOpenAI(chat_responses=[Echo(text="hi")], usage=(1000, 200))
    out = LLMGateway(db, settings, client=fake).chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t")
    assert out == Echo(text="hi")
    row = db.execute("select operation, input_tokens, output_tokens, cost_usd, cached from llm_usage").fetchone()
    assert row["operation"] == "t" and (row["input_tokens"], row["output_tokens"]) == (1000, 200)
    assert float(row["cost_usd"]) == pytest.approx((1000 * 0.10 + 200 * 0.40) / 1e6)
    assert row["cached"] is False


def test_identical_chat_is_cached(db, settings):
    fake = FakeOpenAI(chat_responses=[Echo(text="hi")])
    gw = LLMGateway(db, settings, client=fake)
    gw.chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t")
    assert gw.chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t") == Echo(text="hi")
    assert fake.chat_calls == 1
    cached = db.execute("select cached, cost_usd from llm_usage order by id").fetchall()
    assert [r["cached"] for r in cached] == [False, True] and float(cached[1]["cost_usd"]) == 0


def test_different_model_or_messages_miss_the_cache(db, settings):
    fake = FakeOpenAI(chat_responses=[Echo(text="a"), Echo(text="b"), Echo(text="c")])
    gw = LLMGateway(db, settings, client=fake)
    gw.chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t")
    gw.chat(MSGS, model="gpt-4.1-mini", schema=Echo, operation="t")
    gw.chat([{"role": "user", "content": "y"}], model="gpt-4.1-nano", schema=Echo, operation="t")
    assert fake.chat_calls == 3


def test_refusal_raises_llm_output_error(db, settings):
    gw = LLMGateway(db, settings, client=FakeOpenAI(chat_responses=[None]))
    with pytest.raises(LLMOutputError):
        gw.chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t")


def test_chat_respects_budget(db, settings):
    settings.budget_usd = 0.001
    db.execute(
        "insert into llm_usage (model, operation, input_tokens, cost_usd, cached)"
        " values ('x', 'embed', 0, 0.001, false)"
    )
    db.commit()
    fake = FakeOpenAI(chat_responses=[Echo(text="hi")])
    with pytest.raises(BudgetExceeded):
        LLMGateway(db, settings, client=fake).chat(MSGS, model="gpt-4.1-nano", schema=Echo, operation="t")
    assert fake.chat_calls == 0


def test_settings_model_defaults(settings):
    assert (settings.planner_model, settings.answer_model) == ("gpt-4.1-nano", "gpt-4.1-mini")
    assert settings.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2" and settings.rerank_keep == 4
    assert settings.rerank_min_score == -1.0  # calibrated on real Starlette questions, 2026-10-19
