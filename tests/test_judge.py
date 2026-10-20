import importlib
import os

import pytest
from pydantic import BaseModel

from copilot.llm.gateway import BudgetExceeded, LLMGateway, LLMOutputError
from fakes import FakeOpenAI


class Verdict(BaseModel):
    verdict: str
    reason: str


MSGS = [{"role": "user", "content": "grade this"}]


def test_complete_returns_text_records_cost_and_caches(db, settings):
    fake = FakeOpenAI(usage=(500, 50))
    fake.complete_responses = ["The answer is supported."]
    gw = LLMGateway(db, settings, client=fake)
    assert gw.complete(MSGS, model="gpt-4.1-mini", operation="judge") == "The answer is supported."
    assert gw.complete(MSGS, model="gpt-4.1-mini", operation="judge") == "The answer is supported."
    assert fake.complete_calls == 1  # second identical call: cache hit
    rows = db.execute("select operation, cached, cost_usd from llm_usage order by id").fetchall()
    assert [(r["operation"], r["cached"]) for r in rows] == [("judge", False), ("judge", True)]
    assert float(rows[0]["cost_usd"]) == pytest.approx((500 * 0.40 + 50 * 1.60) / 1e6)


def test_complete_and_chat_caches_do_not_collide(db, settings):
    fake = FakeOpenAI(chat_responses=[Verdict(verdict="yes", reason="r")])
    fake.complete_responses = ["plain text"]
    gw = LLMGateway(db, settings, client=fake)
    gw.chat(MSGS, model="gpt-4.1-mini", schema=Verdict, operation="judge")
    assert gw.complete(MSGS, model="gpt-4.1-mini", operation="judge") == "plain text"


def test_complete_respects_budget(db, settings):
    settings.budget_usd = 0.001
    db.execute("insert into llm_usage (model, operation, input_tokens, cost_usd, cached) values ('x','e',0,0.001,false)")
    db.commit()
    fake = FakeOpenAI()
    fake.complete_responses = ["x"]
    with pytest.raises(BudgetExceeded):
        LLMGateway(db, settings, client=fake).complete(MSGS, model="gpt-4.1-mini", operation="judge")
    assert fake.complete_calls == 0


def test_judge_with_schema_uses_structured_chat(db, settings):
    from copilot.evals.judge import GatewayJudge

    fake = FakeOpenAI(chat_responses=[Verdict(verdict="yes", reason="supported by chunk 2")])
    judge = GatewayJudge(LLMGateway(db, settings, client=fake), "gpt-4.1-mini")
    out = judge.generate("Is claim X supported?", schema=Verdict)
    assert isinstance(out, Verdict) and out.verdict == "yes" and fake.chat_calls == 1
    assert judge.get_model_name() == "gpt-4.1-mini" and judge.load_model() is judge


def test_judge_without_schema_returns_text(db, settings):
    from copilot.evals.judge import GatewayJudge

    fake = FakeOpenAI()
    fake.complete_responses = ["0.8"]
    assert GatewayJudge(LLMGateway(db, settings, client=fake), "gpt-4.1-mini").generate("score?") == "0.8"


def test_judge_falls_back_to_json_text_if_structured_output_is_refused(db, settings):
    # Strict structured outputs may reject a schema at runtime; the judge then asks for plain JSON.
    from copilot.evals.judge import GatewayJudge

    fake = FakeOpenAI(chat_responses=[LLMOutputError("refused")])
    fake.complete_responses = ['Here you go:\n```json\n{"verdict": "no", "reason": "not in context"}\n```']
    out = GatewayJudge(LLMGateway(db, settings, client=fake), "gpt-4.1-mini").generate("grade", schema=Verdict)
    assert out == Verdict(verdict="no", reason="not in context")


def test_judge_async_delegates_to_sync(db, settings):
    import asyncio

    from copilot.evals.judge import GatewayJudge

    fake = FakeOpenAI()
    fake.complete_responses = ["ok"]
    judge = GatewayJudge(LLMGateway(db, settings, client=fake), "gpt-4.1-mini")
    assert asyncio.run(judge.a_generate("x")) == "ok"


def test_importing_evals_turns_off_deepeval_telemetry(monkeypatch):
    monkeypatch.delenv("DEEPEVAL_TELEMETRY_OPT_OUT", raising=False)
    import copilot.evals
    importlib.reload(copilot.evals)
    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "YES"


def test_judge_model_setting(settings):
    assert settings.eval_judge_model == "gpt-4.1-mini"


def test_importing_evals_stops_deepeval_from_loading_dotenv(monkeypatch):
    # DeepEval copies .env into os.environ at import time unless told not to. That would hand every
    # secret in .env to every child process (e.g. the Docker MCP containers get only ONE token by design).
    monkeypatch.delenv("DEEPEVAL_DISABLE_DOTENV", raising=False)
    import copilot.evals
    importlib.reload(copilot.evals)
    assert os.environ["DEEPEVAL_DISABLE_DOTENV"] == "1"


def test_deepeval_pytest_plugin_is_disabled(pytestconfig):
    assert not pytestconfig.pluginmanager.has_plugin("deepeval")
