import re
import uuid
from contextlib import contextmanager

import pytest

from copilot.agent.graph import build_graph, new_turn_input, open_checkpointer
from copilot.agent.nodes import AgentDeps
from copilot.agent.schemas import ActionDraft, AnswerDraft, Plan, RequestedAction
from copilot.config import Settings
from copilot.retrieval.agentic import SubQuery
from copilot.service.runner import InlineExecutor, NotAwaiting, ThreadBusy, ThreadRunner
from fakes import FakeChat, FakeEmbedder, FakeReranker, FakeTools

MAINTAINER = {"approved": True, "by": "k1", "role": "maintainer", "body": None}


def cite_first(messages) -> AnswerDraft:
    first = int(re.search(r'<chunk id="(\d+)"', messages[1]["content"]).group(1))
    return AnswerDraft(answer=f"It times out [{first}].", cited_chunk_ids=[first], insufficient_context=False)


def plan(action: bool = False) -> Plan:
    return Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")],
                requested_action=RequestedAction(kind="comment_on_issue", issue_number=1, instructions="x")
                if action else None)


@pytest.fixture
def env(seeded_db, _test_database):
    with open_checkpointer(_test_database) as cp:
        class Env:
            db = seeded_db
            writes = FakeTools()
            chat = FakeChat()

            def graph(self):
                deps = AgentDeps(conn=seeded_db, llm=self.chat, embedder=FakeEmbedder(), reranker=FakeReranker(),
                                 settings=Settings(_env_file=None, demo_repo="o/r", sandbox_repo="me/sandbox"),
                                 write_tools=self.writes)
                return build_graph(deps, cp, retry_initial_interval=0.01)

            def runner(self, lease_s: float = 60, run_deadline_s: float = 900) -> ThreadRunner:
                @contextmanager
                def factory():
                    yield self.graph()
                return ThreadRunner(factory, _test_database, executor=InlineExecutor(), lease_s=lease_s,
                                    run_deadline_s=run_deadline_s)

        yield Env()


def tid() -> str:
    return uuid.uuid4().hex[:8]


def status(db, thread_id: str) -> str:
    return db.execute("select status from thread_metadata where thread_id = %s", (thread_id,)).fetchone()["status"]


def test_turn_runs_and_records_done(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "What is ERR_TIMEOUT?")
    snap = runner.snapshot(t)
    assert snap["status"] == "done" and snap["answer"]["sources"]
    assert [e["node"] for e in runner.events_since(t, 0)] == ["plan", "retrieve", "live_lookup", "synthesize",
                                                            "verify", "respond"]
    assert runner.events_since(t, 0)[0]["label"] == "Planning searches"
    assert [m["role"] for m in snap["messages"]] == ["user", "assistant"]


def test_events_since_returns_only_new_events(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    assert [e["node"] for e in runner.events_since(t, 4)] == ["verify", "respond"]


def test_busy_thread_rejects_second_turn(env):
    runner, t = env.runner(), tid()
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until) values (%s, 'running', now() + interval '1 minute')", (t,))
    env.db.commit()
    with pytest.raises(ThreadBusy):
        runner.start_turn(t, "q")


def test_failure_records_failed_and_error(env):
    env.chat = FakeChat({"plan": [RuntimeError("boom")]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    snap = runner.snapshot(t)
    assert snap["status"] == "failed" and "RuntimeError" in snap["last_error"]
    assert "boom" in status_row(env.db, t)["last_error"]  # details kept for operators, not clients


def test_failed_thread_can_take_a_new_turn(env):
    env.chat = FakeChat({"plan": [RuntimeError("boom"), plan()], "answer": [cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q1")
    runner.start_turn(t, "q2")
    assert runner.snapshot(t)["status"] == "done"


def test_awaiting_approval_then_decide(env):
    env.chat = FakeChat({"plan": [plan(action=True)], "answer": [cite_first],
                         "action": [ActionDraft(title=None, body="hello")]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "comment on #1")
    snap = runner.snapshot(t)
    assert snap["status"] == "awaiting_approval" and snap["pending_action"]["body"] == "hello"
    with pytest.raises(ThreadBusy):
        runner.start_turn(t, "another question")  # can't start a new turn over a pending action
    runner.decide(t, MAINTAINER)
    snap = runner.snapshot(t)
    assert snap["status"] == "done" and snap["action_result"]["status"] == "executed"
    assert len(env.writes.calls) == 1


def test_decide_when_not_awaiting_raises(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    with pytest.raises(NotAwaiting):
        runner.decide(t, MAINTAINER)  # also what a second "Approve" click hits


def _crash_mid_run(env, t: str) -> None:
    """Simulate a server dying after 'retrieve': checkpoint saved, status still 'running', lease expired."""
    env.graph().invoke(new_turn_input("q"), {"configurable": {"thread_id": t}}, interrupt_after=["retrieve"])
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until)"
                   " values (%s, 'running', now() - interval '1 second')", (t,))
    env.db.commit()


def test_recover_resumes_expired_running_thread_once(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    t = tid()
    _crash_mid_run(env, t)
    runner = env.runner()
    assert runner.recover() == [t]
    assert runner.recover() == []  # lease claimed/finished: never resumed twice
    snap = runner.snapshot(t)
    assert snap["status"] == "done" and snap["answer"]["sources"]
    assert len(env.chat.calls_for("plan")) == 1  # continued from the checkpoint, not restarted


def test_recover_ignores_live_leases(env):
    t = tid()
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until)"
                   " values (%s, 'running', now() + interval '1 minute')", (t,))
    env.db.commit()
    assert env.runner().recover() == []
    assert status(env.db, t) == "running"


def test_snapshot_of_unknown_thread(env):
    snap = env.runner().snapshot("nope1234")
    assert snap["status"] == "idle" and snap["messages"] == [] and snap["answer"] is None


# ---------------------------------------------------------------------------------------------
# Review fixes: heartbeat + fencing, retry after failure, lost questions, deadlines
# ---------------------------------------------------------------------------------------------
import threading
import time

from langgraph.types import Command

from copilot.agent.graph import is_awaiting_approval


def meta(db, thread_id: str) -> dict:
    db.rollback()
    return db.execute("select * from thread_metadata where thread_id = %s", (thread_id,)).fetchone()


def slow_cite(seconds: float, during=None):
    def answer(messages):
        if during:
            during()
        time.sleep(seconds)
        return cite_first(messages)
    return answer


def test_fencing_stale_worker_cannot_overwrite_status(env):
    # Another worker takes the lease mid-run (as if this one had stalled): this run must stop and
    # must not write its status over the new owner's.
    t = tid()

    def takeover():
        with __import__("psycopg").connect(env.runner().database_url) as c:
            c.execute("update thread_metadata set lease_owner = 'someone-else' where thread_id = %s", (t,))

    env.chat = FakeChat({"plan": [plan()], "answer": [slow_cite(0, during=takeover)]})
    runner = env.runner()
    runner.start_turn(t, "q")
    row = meta(env.db, t)
    assert row["status"] == "running" and row["lease_owner"] == "someone-else"
    assert "respond" not in [e["node"] for e in runner.events_since(t, 0)]


def test_heartbeat_keeps_a_slow_node_alive(env):
    # A node that takes longer than the lease must not look dead while it's still working.
    t = tid()
    runner = env.runner(lease_s=0.6)  # heartbeat every 0.2 s
    seen = {}
    threading.Timer(0.9, lambda: seen.setdefault("recovered", runner.recover())).start()
    env.chat = FakeChat({"plan": [plan()], "answer": [slow_cite(1.3)]})
    runner.start_turn(t, "q")
    assert seen["recovered"] == []  # nothing looked dead
    assert meta(env.db, t)["status"] == "done" and len(env.chat.calls_for("answer")) == 1


def test_recover_skips_runs_already_in_this_process(env):
    t = tid()
    runner = env.runner()
    seen = {}

    def expire_and_recover():
        env_db = __import__("psycopg").connect(runner.database_url)
        env_db.execute("update thread_metadata set lease_until = now() - interval '1 second' where thread_id = %s", (t,))
        env_db.commit()
        env_db.close()
        seen["recovered"] = runner.recover()

    env.chat = FakeChat({"plan": [plan()], "answer": [slow_cite(0, during=expire_and_recover)]})
    runner.start_turn(t, "q")
    assert seen["recovered"] == [] and len(env.chat.calls_for("answer")) == 1


def test_deadline_stops_a_hung_run(env):
    t = tid()
    runner = env.runner(lease_s=0.6, run_deadline_s=0.5)
    env.chat = FakeChat({"plan": [plan()], "answer": [slow_cite(1.0)]})
    runner.start_turn(t, "q")
    row = meta(env.db, t)
    assert row["status"] == "failed" and "deadline" in row["last_error"]


def test_failed_run_can_be_retried_from_its_checkpoint(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [RuntimeError("model down"), cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    assert runner.snapshot(t)["status"] == "failed"
    runner.retry(t)
    snap = runner.snapshot(t)
    assert snap["status"] == "done" and snap["answer"]["sources"]
    assert len(env.chat.calls_for("plan")) == 1  # continued from the checkpoint, not restarted


def test_retry_only_from_failed(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    with pytest.raises(NotAwaiting):
        runner.retry(t)


def test_unfinished_approved_action_blocks_new_questions_until_retried(env):
    env.chat = FakeChat({"plan": [plan(action=True)], "answer": [cite_first],
                         "action": [ActionDraft(title=None, body="hello")]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "comment on #1")
    # Approval consumed, but the process died before execute_action ran.
    env.graph().invoke(Command(resume=MAINTAINER), {"configurable": {"thread_id": t}},
                       interrupt_before=["execute_action"])
    env.db.execute("update thread_metadata set status = 'failed' where thread_id = %s", (t,))
    env.db.commit()
    with pytest.raises(ThreadBusy):
        runner.start_turn(t, "unrelated question")  # would silently discard the approved action
    runner.retry(t)
    assert runner.snapshot(t)["action_result"]["status"] == "executed" and len(env.writes.calls) == 1


def test_lost_question_is_replayed_on_recovery(env):
    # The process died after accepting the question but before LangGraph checkpointed it.
    t = tid()
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until, pending_input)"
                   " values (%s, 'running', now() - interval '1 second', %s)", (t, '{"question": "What is ERR_TIMEOUT?"}'))
    env.db.commit()
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    runner = env.runner()
    assert runner.recover() == [t]
    snap = runner.snapshot(t)
    assert snap["status"] == "done" and [m["content"] for m in snap["messages"]][0] == "What is ERR_TIMEOUT?"


def test_recovered_turn_that_already_finished_is_just_marked_done(env):
    # Died after the last checkpoint but before writing 'done': don't run (or pay for) it again.
    t = tid()
    env.chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
    from copilot.agent.graph import run_turn
    run_turn(env.graph(), t, "What is ERR_TIMEOUT?")
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until, pending_input)"
                   " values (%s, 'running', now() - interval '1 second', %s)", (t, '{"question": "What is ERR_TIMEOUT?"}'))
    env.db.commit()
    runner = env.runner()
    assert runner.recover() == [t]
    assert meta(env.db, t)["status"] == "done" and len(env.chat.calls_for("plan")) == 1


def test_new_question_after_a_failed_turn_is_not_confused_with_it(env):
    # Turn 1 failed mid-way; the user asks something else, then the process dies before turn 2's
    # input is checkpointed. Recovery must replay turn 2, not resume turn 1.
    env.chat = FakeChat({"plan": [plan(), plan()], "answer": [RuntimeError("down"), cite_first]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "first question")
    env.db.execute("update thread_metadata set status = 'running', lease_until = now() - interval '1 second',"
                   " pending_input = %s where thread_id = %s", ('{"question": "second question"}', t))
    env.db.commit()
    runner.recover()
    msgs = runner.snapshot(t)["messages"]
    assert msgs[-2]["content"] == "second question" and msgs[-1]["role"] == "assistant"


def test_decide_resets_progress_events(env):
    env.chat = FakeChat({"plan": [plan(action=True)], "answer": [cite_first],
                         "action": [ActionDraft(title=None, body="hello")]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "comment on #1")
    runner.decide(t, MAINTAINER)
    assert runner.events_since(t, 0)[0]["node"] == "execute_action"


def test_snapshot_hides_error_details(env):
    env.chat = FakeChat({"plan": [RuntimeError("secret internal detail")]})
    runner, t = env.runner(), tid()
    runner.start_turn(t, "q")
    snap = runner.snapshot(t)
    assert "secret internal detail" not in snap["last_error"] and "RuntimeError" in snap["last_error"]
    assert "secret internal detail" in meta(env.db, t)["last_error"]  # still there for operators


def status_row(db, thread_id: str) -> dict:
    return meta(db, thread_id)


def test_close_waits_for_in_flight_runs(env, _test_database):
    # Review #6: shutdown must let runs finish BEFORE the MCP clients they use are closed.
    from concurrent.futures import ThreadPoolExecutor

    env.chat = FakeChat({"plan": [plan()], "answer": [slow_cite(0.5)]})

    @contextmanager
    def factory():
        yield env.graph()

    runner = ThreadRunner(factory, _test_database, executor=ThreadPoolExecutor(max_workers=1))
    t = tid()
    runner.start_turn(t, "q")
    runner.close()  # returns only after the run is done
    assert meta(env.db, t)["status"] == "done"
