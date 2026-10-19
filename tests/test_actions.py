import re
import uuid

import pytest

from copilot.agent.graph import ThreadAwaitingApproval, build_graph, open_checkpointer, resume_turn, run_turn
from copilot.agent.nodes import AgentDeps, execute_action_node
from copilot.agent.schemas import ActionDraft, AnswerDraft, Plan, RequestedAction
from copilot.config import Settings
from copilot.retrieval.agentic import SubQuery
from fakes import FakeChat, FakeEmbedder, FakeReranker, FakeTools

SANDBOX = "me/sandbox"
MAINTAINER = {"approved": True, "by": "k1", "role": "maintainer", "body": None}


def settings(sandbox: str = SANDBOX) -> Settings:
    return Settings(_env_file=None, demo_repo="o/r", sandbox_repo=sandbox)


def plan_with_action(kind: str = "comment_on_issue", number: int | None = 1) -> Plan:
    return Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")],
                requested_action=RequestedAction(kind=kind, issue_number=number,
                                                 instructions="summarize the timeout fix"))


def cite_first(messages) -> AnswerDraft:
    first = int(re.search(r'<chunk id="(\d+)"', messages[1]["content"]).group(1))
    return AnswerDraft(answer=f"It times out [{first}].", cited_chunk_ids=[first], insufficient_context=False)


def chat(**overrides) -> FakeChat:
    responses = {"plan": [plan_with_action()], "answer": [cite_first],
                 "action": [ActionDraft(title=None, body="Timeout summary for the team.")]}
    responses.update(overrides)
    return FakeChat(responses)


@pytest.fixture
def env(seeded_db, _test_database):
    with open_checkpointer(_test_database) as cp:
        class Env:
            db = seeded_db
            writes = FakeTools()

            def graph(self, fake_chat=None, sandbox=SANDBOX, write_tools="default"):
                tools = self.writes if write_tools == "default" else write_tools
                deps = AgentDeps(conn=seeded_db, llm=fake_chat or FakeChat(), embedder=FakeEmbedder(),
                                 reranker=FakeReranker(), settings=settings(sandbox), write_tools=tools)
                return build_graph(deps, cp, retry_initial_interval=0.01)

        yield Env()


def tid() -> str:
    return uuid.uuid4().hex[:8]


def action_rows(db) -> list[dict]:
    return db.execute("select status, target_repo, decided_by from actions order by decided_at").fetchall()


def test_requested_comment_pauses_for_approval(env):
    res = run_turn(env.graph(chat()), tid(), "Comment on #1 summarizing the timeout fix")
    assert res["status"] == "awaiting_approval"
    p = res["pending_action"]
    assert (p["kind"], p["repo"], p["issue_number"]) == ("comment_on_issue", SANDBOX, 1)
    assert p["body"] == "Timeout summary for the team."
    assert res["sources"]  # the verified answer is already available while waiting
    assert env.writes.calls == []  # nothing written before approval


def test_pending_action_survives_restart_and_executes(env):
    t = tid()
    run_turn(env.graph(chat()), t, "Comment on #1 summarizing the timeout fix")
    res = resume_turn(env.graph(), t, MAINTAINER)  # brand-new graph object = restarted process
    assert res["status"] == "done"
    name, args = env.writes.calls[0]
    assert name == "add_issue_comment"
    assert {k: args[k] for k in ("owner", "repo", "issue_number")} == {"owner": "me", "repo": "sandbox", "issue_number": 1}
    assert args["body"].startswith("Timeout summary for the team.\n\n<!-- copilot:action:")  # + hidden marker
    assert res["action_result"]["status"] == "executed" and res["action_result"]["url"]
    assert "Posted" in res["text"]
    assert [r["status"] for r in action_rows(env.db)] == ["executed"]


def test_create_issue_uses_issue_write(env):
    c = chat(plan=[plan_with_action("create_issue", None)],
             action=[ActionDraft(title="Timeouts", body="Details")])
    t = tid()
    run_turn(env.graph(c), t, "Open an issue about timeouts")
    resume_turn(env.graph(), t, MAINTAINER)
    name, args = env.writes.calls[0]
    assert name == "issue_write" and args["method"] == "create" and args["title"] == "Timeouts"


def test_edited_body_is_used(env):
    t = tid()
    run_turn(env.graph(chat()), t, "Comment on #1")
    resume_turn(env.graph(), t, {**MAINTAINER, "body": "Edited by a human."})
    assert env.writes.calls[0][1]["body"].startswith("Edited by a human.\n\n<!-- copilot:action:")


def test_reject_executes_nothing(env):
    t = tid()
    run_turn(env.graph(chat()), t, "Comment on #1")
    res = resume_turn(env.graph(), t, {"approved": False, "by": "k1", "role": "maintainer", "body": None})
    assert env.writes.calls == [] and res["action_result"]["status"] == "rejected"
    assert "Not posted" in res["text"] and [r["status"] for r in action_rows(env.db)] == ["rejected"]


def test_reader_cannot_execute(env):
    t = tid()
    run_turn(env.graph(chat()), t, "Comment on #1")
    res = resume_turn(env.graph(), t, {**MAINTAINER, "role": "reader"})
    assert env.writes.calls == [] and res["action_result"]["status"] == "refused"


def test_wrong_repo_refused_in_tool_layer(env):
    t = tid()
    graph = env.graph(chat())
    run_turn(graph, t, "Comment on #1")
    config = {"configurable": {"thread_id": t}}
    pending = graph.get_state(config).values["pending_action"]
    graph.update_state(config, {"pending_action": {**pending, "repo": "Kludex/starlette"}})  # tampered
    res = resume_turn(env.graph(), t, MAINTAINER)
    assert env.writes.calls == [] and res["action_result"]["status"] == "refused"


def test_executes_at_most_once(env):
    pending = {"kind": "comment_on_issue", "repo": SANDBOX, "issue_number": 1, "title": None,
               "body": "hi", "idempotency_key": f"{tid()}:1"}
    deps = AgentDeps(conn=env.db, llm=FakeChat(), embedder=FakeEmbedder(), reranker=FakeReranker(),
                     settings=settings(), write_tools=env.writes)
    state = {"pending_action": pending, "approval": MAINTAINER}
    first = execute_action_node(state, deps)
    second = execute_action_node(state, deps)  # double-click / retry / race
    assert len(env.writes.calls) == 1
    assert first["action_result"]["status"] == second["action_result"]["status"] == "executed"
    assert second["action_result"]["url"] == first["action_result"]["url"]


@pytest.mark.parametrize("raw,url", [
    # GitHub MCP v2.0.2 returns MinimalResponse {"id", "url"} (found on the real run: url was None).
    ({"id": "5", "url": "https://github.com/me/sandbox/issues/1#issuecomment-5"},
     "https://github.com/me/sandbox/issues/1#issuecomment-5"),
    ({"Single": {"id": "5", "url": "https://github.com/me/sandbox/issues/1#issuecomment-5"}},
     "https://github.com/me/sandbox/issues/1#issuecomment-5"),
    ({"comment": {"id": "5", "url": "https://github.com/me/sandbox/issues/1#issuecomment-5"}},
     "https://github.com/me/sandbox/issues/1#issuecomment-5"),
    ({"html_url": "https://github.com/me/sandbox/issues/9", "url": "https://api.github.com/repos/me/sandbox/issues/9"},
     "https://github.com/me/sandbox/issues/9"),
    ({"url": "https://api.github.com/repos/me/sandbox/issues/9"}, None),  # API link is not a page to open
    ("not json", None),
])
def test_result_url_handles_real_response_shapes(raw, url):
    from copilot.agent.nodes import result_url
    assert result_url(raw) == url


def test_write_failure_is_recorded(env):
    t = tid()
    run_turn(env.graph(chat()), t, "Comment on #1")
    res = resume_turn(env.graph(write_tools=FakeTools(fail=True)), t, MAINTAINER)
    assert res["action_result"]["status"] == "failed" and [r["status"] for r in action_rows(env.db)] == ["failed"]


def test_no_sandbox_configured_skips_interrupt(env):
    res = run_turn(env.graph(chat(), sandbox=""), tid(), "Comment on #1")
    assert res["status"] == "done" and res["action_result"]["status"] == "unavailable"
    assert env.writes.calls == []


def test_new_question_while_awaiting_is_refused(env):
    t = tid()
    graph = env.graph(chat())
    run_turn(graph, t, "Comment on #1")
    with pytest.raises(ThreadAwaitingApproval):
        run_turn(graph, t, "another question")


def test_assistant_messages_keep_their_sources_and_action(env):
    # The UI renders citations for EVERY past answer, not just the latest one.
    t = tid()
    graph = env.graph(chat())
    run_turn(graph, t, "Comment on #1")
    resume_turn(env.graph(), t, MAINTAINER)
    last = graph.get_state({"configurable": {"thread_id": t}}).values["messages"][-1]
    assert last["role"] == "assistant" and last["sources"] and last["sources"][0]["number"] == 11
    assert last["text"].startswith("It times out [") and "Posted" not in last["text"]  # answer only
    assert "Posted to GitHub" in last["content"]  # the history the planner sees still mentions it
    assert last["action"]["status"] == "executed" and last["action"]["url"]


def test_plain_question_has_no_action(env):
    c = chat(plan=[Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")])])
    res = run_turn(env.graph(c), tid(), "What is ERR_TIMEOUT?")
    assert res["status"] == "done" and res["pending_action"] is None and res["action_result"] is None


# ---------------------------------------------------------------------------------------------
# Review fixes: reconciliation of ambiguous outcomes, markers, unique keys, empty edits
# ---------------------------------------------------------------------------------------------
from copilot.tools.mcp_client import ToolError


def node_deps(env, write_tools=None, tools=None, sandbox=SANDBOX) -> AgentDeps:
    return AgentDeps(conn=env.db, llm=FakeChat(), embedder=FakeEmbedder(), reranker=FakeReranker(),
                     settings=settings(sandbox), tools=tools, write_tools=write_tools or env.writes)


def pending(key=None, kind="comment_on_issue", body="hi", title=None) -> dict:
    return {"kind": kind, "repo": SANDBOX, "issue_number": 1 if kind == "comment_on_issue" else None,
            "title": title, "body": body, "idempotency_key": key or f"{tid()}:{uuid.uuid4().hex}"}


def stale_row(env, p: dict, status: str = "pending", age_s: int = 3600) -> None:
    env.db.execute(
        "insert into actions (idempotency_key, thread_id, kind, target_repo, payload, status, decided_at)"
        " values (%s, 't', %s, %s, '{}', %s, now() - make_interval(secs => %s))",
        (p["idempotency_key"], p["kind"], p["repo"], status, age_s))
    env.db.commit()


def test_posted_body_carries_a_hidden_marker(env):
    p = pending()
    execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env))
    body = env.writes.calls[0][1]["body"]
    assert body.startswith("hi") and f"<!-- copilot:action:{p['idempotency_key']} -->" in body


def test_marker_survives_an_edited_body(env):
    p = pending()
    execute_action_node({"pending_action": p, "approval": {**MAINTAINER, "body": "edited"}}, node_deps(env))
    body = env.writes.calls[0][1]["body"]
    assert body.startswith("edited") and p["idempotency_key"] in body


def test_idempotency_keys_are_unique_per_proposal(env):
    keys = []
    for _ in range(2):
        t = tid()
        run_turn(env.graph(chat()), t, "Comment on #1")
        keys.append(env.graph().get_state({"configurable": {"thread_id": t}}).values["pending_action"]["idempotency_key"])
    assert keys[0] != keys[1] and len(keys[0].split(":")[1]) == 32


def test_empty_edited_body_is_refused(env):
    out = execute_action_node({"pending_action": pending(), "approval": {**MAINTAINER, "body": "   "}}, node_deps(env))
    assert out["action_result"]["status"] == "refused" and env.writes.calls == []


def test_timeout_is_recorded_as_unknown_not_failed(env):
    # A timed-out call may still have been executed by GitHub: don't claim it failed.
    writes = FakeTools(write_error=ToolError("add_issue_comment timed out after 15s", uncertain=True))
    out = execute_action_node({"pending_action": pending(), "approval": MAINTAINER}, node_deps(env, writes))
    assert out["action_result"]["status"] == "unknown"
    assert env.db.execute("select status from actions").fetchone()["status"] == "unknown"


def test_definite_tool_error_is_failed(env):
    writes = FakeTools(write_error=ToolError("add_issue_comment returned an error: 404", uncertain=False))
    out = execute_action_node({"pending_action": pending(), "approval": MAINTAINER}, node_deps(env, writes))
    assert out["action_result"]["status"] == "failed"


def test_unexpected_exception_is_unknown(env):
    writes = FakeTools(write_error=RuntimeError("connection reset"))
    out = execute_action_node({"pending_action": pending(), "approval": MAINTAINER}, node_deps(env, writes))
    assert out["action_result"]["status"] == "unknown"


def test_stale_pending_is_reconciled_when_the_comment_exists(env):
    # Crash after GitHub posted but before we recorded it: find our marker, don't post again.
    p = pending()
    stale_row(env, p)
    reads = FakeTools()
    reads.comments[1] = [{"body": f"hi\n\n<!-- copilot:action:{p['idempotency_key']} -->",
                          "html_url": "https://github.com/me/sandbox/issues/1#issuecomment-77"}]
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=reads))
    assert out["action_result"] == {"status": "executed", "url": "https://github.com/me/sandbox/issues/1#issuecomment-77"}
    assert env.writes.calls == []
    assert env.db.execute("select status from actions").fetchone()["status"] == "executed"


def test_stale_pending_without_comment_is_posted_once(env):
    # Crash after claiming but before calling GitHub: nothing was posted, so post now.
    p = pending()
    stale_row(env, p)
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=FakeTools()))
    assert out["action_result"]["status"] == "executed" and len(env.writes.calls) == 1


def test_stale_unknown_is_reconciled_too(env):
    p = pending()
    stale_row(env, p, status="unknown")
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=FakeTools()))
    assert out["action_result"]["status"] == "executed" and len(env.writes.calls) == 1


def test_fresh_pending_is_left_alone(env):
    # Another worker may be mid-call right now: reconciling could race it.
    p = pending()
    stale_row(env, p, age_s=0)
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=FakeTools()))
    assert out["action_result"]["status"] == "pending" and env.writes.calls == []


def test_stale_issue_creation_cannot_be_verified_so_it_is_unknown(env):
    p = pending(kind="create_issue", title="T")
    stale_row(env, p)
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=FakeTools()))
    assert out["action_result"]["status"] == "unknown" and env.writes.calls == []


def test_reconcile_without_read_access_is_unknown(env):
    p = pending()
    stale_row(env, p)
    out = execute_action_node({"pending_action": p, "approval": MAINTAINER}, node_deps(env, tools=None))
    assert out["action_result"]["status"] == "unknown" and env.writes.calls == []
