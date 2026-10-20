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


def test_trace_report_command(cli_env, monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 10, 20, tzinfo=timezone.utc)
    rows = [{"name": "agent_turn", "span_kind": "CHAIN", "trace_id": "a", "start_time": t0,
             "end_time": t0 + timedelta(seconds=3), "attributes": {}},
            {"name": "llm.answer", "span_kind": "LLM", "trace_id": "a", "start_time": t0,
             "end_time": t0 + timedelta(seconds=2), "attributes": {"copilot.cost_usd": 0.001, "copilot.cache_hit": False}}]
    seen = {}
    monkeypatch.setattr(cli, "fetch_spans", lambda s, since_hours: seen.setdefault("h", since_hours) and rows)
    out = tmp_path / "report.md"
    r = runner.invoke(cli.app, ["trace-report", "--since-hours", "6", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert seen["h"] == 6 and "1 turns" in r.output and "llm.answer" in out.read_text()


@pytest.fixture
def evals_env(cli_env, monkeypatch):
    from fakes import FakeChat
    from test_eval_dataset import drafter

    monkeypatch.setenv("COPILOT_DEMO_REPO", "o/r")
    get_settings.cache_clear()
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: FakeChat({"eval_draft": [drafter] * 30}))


def test_evals_draft_writes_drafts_plus_seed_cases(evals_env, tmp_path):
    from copilot.evals.dataset import load_golden

    out = tmp_path / "golden_draft.jsonl"
    r = runner.invoke(cli.app, ["evals", "draft", "--docs", "2", "--history", "2", "--multi", "1",
                                "--min-chars", "0", "--out", str(out)])
    assert r.exit_code == 0, r.output
    cases = load_golden(out)
    cats = [c.category for c in cases]
    assert cats.count("docs") == 2 and "unanswerable" in cats and "adversarial" in cats
    assert "review" in r.output.lower()


def test_evals_draft_refuses_to_overwrite_golden(evals_env, tmp_path):
    r = runner.invoke(cli.app, ["evals", "draft", "--out", str(tmp_path / "golden.jsonl")])
    assert r.exit_code != 0 and "reviewed" in r.output


def test_evals_check_reports_problems(evals_env, tmp_path):
    from copilot.evals.dataset import EvalCase, write_cases

    good = EvalCase(id="docs-01", category="docs", question="q", expected_sources=["doc:docs/config.md"],
                    reference_answer="a", in_ci=True)
    path = tmp_path / "golden.jsonl"
    write_cases(path, [good])
    r = runner.invoke(cli.app, ["evals", "check", "--path", str(path)])
    assert r.exit_code == 0 and "docs: 1" in r.output and "in CI subset: 1" in r.output
    write_cases(path, [good, good.model_copy(update={"id": "docs-02", "expected_sources": ["doc:gone.md"]})])
    r = runner.invoke(cli.app, ["evals", "check", "--path", str(path)])
    assert r.exit_code == 1 and "doc:gone.md" in r.output


def test_evals_retrieval_command(evals_env, monkeypatch, tmp_path):
    from copilot.evals.dataset import EvalCase, write_cases
    from fakes import FakeReranker

    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    path, out = tmp_path / "golden.jsonl", tmp_path / "retrieval.md"
    write_cases(path, [EvalCase(id="h1", category="history", question="ERR_TIMEOUT raised by the client",
                                expected_sources=["issue:11"], reference_answer="r")])
    r = runner.invoke(cli.app, ["evals", "retrieval", "--path", str(path), "--out", str(out), "--k", "3"])
    assert r.exit_code == 0, r.output
    assert "| sparse |" in r.output and "hit rate@3" in out.read_text()


def test_evals_retrieval_explains_missing_golden_file(evals_env, tmp_path):
    r = runner.invoke(cli.app, ["evals", "retrieval", "--path", str(tmp_path / "golden.jsonl")])
    assert r.exit_code != 0 and "golden_draft.jsonl" in r.output


@pytest.fixture
def run_env(cli_env, monkeypatch, tmp_path):
    from copilot.agent.schemas import AnswerDraft, Plan
    from copilot.evals.dataset import EvalCase, write_cases
    from copilot.retrieval.agentic import SubQuery
    from fakes import FakeChat, FakeReranker

    monkeypatch.setenv("COPILOT_DEMO_REPO", "o/r")
    get_settings.cache_clear()
    plan = Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")])
    chat = FakeChat({"plan": [plan] * 10, "answer": [_cite_first] * 10})
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: chat)
    monkeypatch.setattr(cli, "build_reranker", lambda s: FakeReranker())
    golden = tmp_path / "golden.jsonl"
    write_cases(golden, [
        EvalCase(id="h1", category="history", question="What does ERR_TIMEOUT mean?", expected_sources=["issue:11"],
                 reference_answer="The client timed out.", in_ci=True),
        EvalCase(id="h2", category="history", question="Not in CI", expected_sources=["issue:11"],
                 reference_answer="x"),
    ])
    return golden, tmp_path


def test_evals_run_no_judge_passes_gate_and_writes_report(run_env):
    golden, tmp = run_env
    thresholds = tmp / "t.json"
    thresholds.write_text('{"hit_rate@5": 0.5}')
    out = tmp / "answers.md"
    r = runner.invoke(cli.app, ["evals", "run", "--path", str(golden), "--ci-only", "--no-judge",
                                "--thresholds", str(thresholds), "--out", str(out)])
    assert r.exit_code == 0, r.output
    report = out.read_text()
    assert "Quality gate: PASSED" in report and "hit_rate@5" in report and "| cases | 1 |" in report


def test_evals_run_fails_the_gate_with_exit_code_1(run_env):
    golden, tmp = run_env
    thresholds = tmp / "t.json"
    thresholds.write_text('{"hit_rate@5": 1.01}')
    r = runner.invoke(cli.app, ["evals", "run", "--path", str(golden), "--no-judge", "--thresholds", str(thresholds)])
    assert r.exit_code == 1 and "FAILED" in r.output


def test_evals_run_without_openai_key_runs_retrieval_gate_only(run_env, monkeypatch):
    # Fork PRs get no secrets: CI must still gate retrieval (cached embeddings) instead of crashing.
    golden, tmp = run_env
    monkeypatch.setenv("COPILOT_OPENAI_API_KEY", "")
    monkeypatch.setattr(cli, "build_chat", lambda conn, s: pytest.fail("no key: the agent must not run"))
    get_settings.cache_clear()
    out = tmp / "report.md"
    r = runner.invoke(cli.app, ["evals", "run", "--path", str(golden), "--ci-only", "--out", str(out)])
    assert r.exit_code == 0, r.output
    report = out.read_text()
    assert "LLM-judged metrics skipped" in report and "Quality gate: PASSED" in report and "hit_rate@5" in report


def test_evals_run_without_key_still_fails_a_bad_retrieval_gate(run_env, monkeypatch):
    golden, tmp = run_env
    monkeypatch.setenv("COPILOT_OPENAI_API_KEY", "")
    get_settings.cache_clear()
    thresholds = tmp / "t.json"
    thresholds.write_text('{"hit_rate@5": 1.01, "faithfulness": 0.85}')
    r = runner.invoke(cli.app, ["evals", "run", "--path", str(golden), "--thresholds", str(thresholds)])
    assert r.exit_code == 1 and "hit_rate@5" in r.output and "faithfulness 0" not in r.output


def test_spend_raw_prints_only_the_number(cli_env):
    r = runner.invoke(cli.app, ["spend", "--raw"])
    assert r.exit_code == 0 and float(r.output.strip()) == 0.0


def test_snapshot_export_then_import_needs_confirmation(cli_env, tmp_path):
    path = tmp_path / "eval-snapshot.tar.gz"
    r = runner.invoke(cli.app, ["evals", "snapshot", "export", str(path)])
    assert r.exit_code == 0 and path.exists() and "documents" in r.output, r.output

    r = runner.invoke(cli.app, ["evals", "snapshot", "import", str(path)], input="n\n")
    assert r.exit_code != 0  # declined: nothing replaced

    r = runner.invoke(cli.app, ["evals", "snapshot", "import", str(path), "--yes"])
    assert r.exit_code == 0 and "Imported" in r.output, r.output
    assert cli_env.execute("select count(*) as n from documents").fetchone()["n"] == 6
