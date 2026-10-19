import pytest
from typer.testing import CliRunner

from copilot import cli
from copilot.config import get_settings
from fakes import FakeEmbedder

runner = CliRunner()


@pytest.fixture
def cli_env(seeded_db, settings, monkeypatch):
    monkeypatch.setenv("COPILOT_DATABASE_URL", settings.database_url)
    monkeypatch.setenv("COPILOT_BUDGET_USD", "8")
    monkeypatch.setattr(cli, "build_embedder", lambda conn, s: FakeEmbedder())
    # Never start real Docker/MCP from tests (Settings() would read real tokens from .env).
    from contextlib import nullcontext
    monkeypatch.setattr(cli, "build_tools", lambda s, write=False: nullcontext(None))
    get_settings.cache_clear()
    yield seeded_db
    get_settings.cache_clear()


def test_search_command_prints_results(cli_env):
    r = runner.invoke(cli.app, ["search", "ERR_TIMEOUT", "--mode", "sparse", "-k", "2"])
    assert r.exit_code == 0, r.output
    assert "ERR_TIMEOUT" in r.output and "[issue #11]" in r.output
    assert "https://github.com/o/r/" in r.output


def test_search_type_and_state_filters(cli_env):
    r = runner.invoke(cli.app, ["search", "background tasks", "--type", "pr", "--state", "merged"])
    assert r.exit_code == 0, r.output
    assert "[pr #12]" in r.output and "[issue" not in r.output


def test_search_no_results_message(cli_env):
    r = runner.invoke(cli.app, ["search", "zzzqqq", "--mode", "sparse"])
    assert r.exit_code == 0 and "No results" in r.output


def test_spend_command(cli_env):
    r = runner.invoke(cli.app, ["spend"])
    assert r.exit_code == 0 and "of $8.00" in r.output


def test_sync_full_flag_is_passed_through(cli_env, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "sync_repo", lambda *a, full=False, **kw: seen.setdefault("full", full) and {})
    monkeypatch.setenv("COPILOT_DEMO_REPO", "o/r")
    get_settings.cache_clear()
    runner.invoke(cli.app, ["sync", "--full"])
    assert seen["full"] is True


def _cite_first(messages):
    import re

    from copilot.agent.schemas import AnswerDraft

    first = int(re.search(r'<chunk id="(\d+)"', messages[1]["content"]).group(1))
    return AnswerDraft(answer=f"It times out [{first}].", cited_chunk_ids=[first], insufficient_context=False)


@pytest.fixture
def action_env(cli_env, monkeypatch):
    from contextlib import nullcontext

    from copilot.agent.schemas import ActionDraft, Plan, RequestedAction
    from copilot.retrieval.agentic import SubQuery
    from fakes import FakeChat, FakeReranker, FakeTools

    monkeypatch.setenv("COPILOT_SANDBOX_REPO", "me/sandbox")
    get_settings.cache_clear()
    plan = Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")],
                requested_action=RequestedAction(kind="comment_on_issue", issue_number=1, instructions="x"))
    chat = FakeChat({"plan": [plan], "answer": [_cite_first], "action": [ActionDraft(title=None, body="hello")]})
    writes = FakeTools()
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: chat)
    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    monkeypatch.setattr(cli, "build_tools", lambda s, write=False: nullcontext(writes if write else FakeTools()))
    return writes


def test_ask_pauses_for_approval_then_approve_executes_once(action_env):
    r = runner.invoke(cli.app, ["ask", "Comment on #1 about timeouts", "--thread", "act1"])
    assert r.exit_code == 0, r.output
    assert "Awaiting approval" in r.output and "me/sandbox" in r.output and "hello" in r.output
    assert action_env.calls == []
    r = runner.invoke(cli.app, ["ask", "--thread", "act1", "--approve"])
    assert r.exit_code == 0, r.output
    assert "Posted to GitHub" in r.output and len(action_env.calls) == 1
    r = runner.invoke(cli.app, ["ask", "--thread", "act1", "--approve"])
    assert r.exit_code == 1 and "Nothing is waiting for approval" in r.output
    assert len(action_env.calls) == 1


def test_ask_reject(action_env):
    runner.invoke(cli.app, ["ask", "Comment on #1", "--thread", "act2"])
    r = runner.invoke(cli.app, ["ask", "--thread", "act2", "--reject"])
    assert r.exit_code == 0 and "Not posted" in r.output and action_env.calls == []


def test_ask_needs_question_or_decision(cli_env):
    r = runner.invoke(cli.app, ["ask"])
    assert r.exit_code != 0


@pytest.fixture
def ask_env(cli_env, monkeypatch):
    from copilot.agent.schemas import Plan
    from copilot.retrieval.agentic import SubQuery
    from fakes import FakeChat, FakeReranker

    plan = Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")])
    chat = FakeChat({"plan": [plan, plan], "answer": [_cite_first, _cite_first]})
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: chat)
    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    return chat


def test_ask_prints_answer_sources_and_thread(ask_env):
    r = runner.invoke(cli.app, ["ask", "What does ERR_TIMEOUT mean?"])
    assert r.exit_code == 0, r.output
    assert "It times out [" in r.output and "Sources:" in r.output
    assert "[issue #11]" in r.output and "https://github.com/o/r/issue/11" in r.output
    assert "thread: " in r.output and "this question: $" in r.output


def test_ask_continues_a_thread(ask_env):
    runner.invoke(cli.app, ["ask", "What does ERR_TIMEOUT mean?", "--thread", "abc123"])
    r = runner.invoke(cli.app, ["ask", "And how do I fix it?", "--thread", "abc123"])
    assert r.exit_code == 0 and "thread: abc123" in r.output
    second_plan = " ".join(m["content"] for m in ask_env.calls_for("plan")[1]["messages"])
    assert "What does ERR_TIMEOUT mean?" in second_plan


def test_ask_budget_exceeded_is_friendly(cli_env, monkeypatch):
    from copilot.llm.gateway import BudgetExceeded
    from fakes import FakeChat, FakeReranker

    monkeypatch.setattr(cli, "build_chat", lambda conn, s: FakeChat({"plan": [BudgetExceeded("ceiling")]}))
    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1 and "budget ceiling reached" in r.output


@pytest.mark.parametrize("make_error", [
    lambda: __import__("copilot.llm.gateway", fromlist=["LLMOutputError"]).LLMOutputError("refused"),
    lambda: __import__("openai").APIConnectionError(request=__import__("httpx").Request("POST", "https://x")),
])
def test_ask_model_errors_are_friendly(cli_env, monkeypatch, make_error):
    from fakes import FakeChat, FakeReranker

    errors = [make_error() for _ in range(3)]  # LLMOutputError is retried 3x by the graph
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: FakeChat({"plan": errors}))
    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    r = runner.invoke(cli.app, ["ask", "q"])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert "couldn't get an answer from the model" in r.output and "thread: " in r.output


def test_init_db_is_safe_to_rerun(cli_env):
    r = runner.invoke(cli.app, ["init-db"])
    assert r.exit_code == 0 and "Schema ready" in r.output
