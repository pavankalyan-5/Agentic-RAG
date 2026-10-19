import json
import re
import uuid
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from copilot.agent.graph import build_graph, new_turn_input, open_checkpointer
from copilot.agent.nodes import AgentDeps
from copilot.agent.schemas import ActionDraft, AnswerDraft, Plan, RequestedAction
from copilot.config import Settings
from copilot.retrieval.agentic import SubQuery
from copilot.service.api import create_app
from copilot.service.runner import InlineExecutor, ThreadRunner
from fakes import FakeChat, FakeEmbedder, FakeReranker, FakeTools

M = {"X-API-Key": "m-key"}  # maintainer
R = {"X-API-Key": "r-key"}  # reader


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
            chat = FakeChat({"plan": [plan()], "answer": [cite_first]})
            settings = Settings(_env_file=None, demo_repo="o/r", sandbox_repo="me/sandbox",
                                api_keys={"m-key": "maintainer", "r-key": "reader", "r2-key": "reader"})

            def graph(self):
                deps = AgentDeps(conn=seeded_db, llm=self.chat, embedder=FakeEmbedder(), reranker=FakeReranker(),
                                 settings=self.settings, write_tools=self.writes)
                return build_graph(deps, cp, retry_initial_interval=0.01)

            def client(self, **settings_overrides) -> TestClient:
                @contextmanager
                def factory():
                    yield self.graph()
                runner = ThreadRunner(factory, _test_database, executor=InlineExecutor())
                return TestClient(create_app(runner, self.settings.model_copy(update=settings_overrides)))

        yield Env()


def new_thread(client) -> str:
    r = client.post("/api/v1/threads", headers=R)
    assert r.status_code == 201
    return r.json()["thread_id"]


def test_health_needs_no_key(env):
    with env.client() as c:
        assert c.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_requires_valid_api_key(env, headers):
    with env.client() as c:
        assert c.post("/api/v1/threads", headers=headers).status_code == 401


def test_ask_and_get_answer(env):
    with env.client() as c:
        t = new_thread(c)
        r = c.post(f"/api/v1/threads/{t}/messages", json={"question": "What is ERR_TIMEOUT?"}, headers=R)
        assert r.status_code == 202
        snap = c.get(f"/api/v1/threads/{t}", headers=R).json()
        assert snap["status"] == "done" and snap["answer"]["sources"][0]["number"] == 11


def test_question_length_validated(env):
    with env.client() as c:
        t = new_thread(c)
        for q in ("", "x" * 2001):
            assert c.post(f"/api/v1/threads/{t}/messages", json={"question": q}, headers=R).status_code == 422


def test_thread_id_format_validated(env):
    with env.client() as c:
        assert c.get("/api/v1/threads/not a valid id!", headers=R).status_code == 422


def test_busy_returns_409(env):
    with env.client() as c:
        t = new_thread(c)
        env.db.execute("update thread_metadata set status = 'running', lease_until = now() + interval '1 minute'"
                       " where thread_id = %s", (t,))
        env.db.commit()
        r = c.post(f"/api/v1/threads/{t}/messages", json={"question": "q"}, headers=R)
        assert r.status_code == 409


def _awaiting(env, c) -> str:
    env.chat = FakeChat({"plan": [plan(action=True)], "answer": [cite_first],
                         "action": [ActionDraft(title=None, body="hello")]})
    t = new_thread(c)
    c.post(f"/api/v1/threads/{t}/messages", json={"question": "comment on #1"}, headers=R)
    assert c.get(f"/api/v1/threads/{t}", headers=R).json()["status"] == "awaiting_approval"
    return t


def test_reader_cannot_approve(env):
    with env.client() as c:
        t = _awaiting(env, c)
        r = c.post(f"/api/v1/threads/{t}/approve", json={"approved": True}, headers=R)
        assert r.status_code == 403 and env.writes.calls == []
        assert c.get(f"/api/v1/threads/{t}", headers=R).json()["status"] == "awaiting_approval"


def test_maintainer_approves_once_second_click_is_409(env):
    with env.client() as c:
        t = _awaiting(env, c)
        assert c.post(f"/api/v1/threads/{t}/approve", json={"approved": True}, headers=M).status_code == 202
        assert c.post(f"/api/v1/threads/{t}/approve", json={"approved": True}, headers=M).status_code == 409
        assert len(env.writes.calls) == 1
        snap = c.get(f"/api/v1/threads/{t}", headers=M).json()
        assert snap["action_result"]["status"] == "executed"
        row = env.db.execute("select decided_by from actions").fetchone()
        assert row["decided_by"] and "m-key" not in row["decided_by"]  # key fingerprint, never the key


def test_edited_body_passed_through(env):
    with env.client() as c:
        t = _awaiting(env, c)
        c.post(f"/api/v1/threads/{t}/approve", json={"approved": True, "body": "edited"}, headers=M)
        assert env.writes.calls[0][1]["body"].startswith("edited\n\n<!-- copilot:action:")


def test_events_stream_lists_nodes_then_status(env):
    with env.client() as c:
        t = new_thread(c)
        c.post(f"/api/v1/threads/{t}/messages", json={"question": "q"}, headers=R)
        with c.stream("GET", f"/api/v1/threads/{t}/events", headers=R) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(line[len("data: "):]) for line in r.iter_lines() if line.startswith("data: ")]
        assert [e["node"] for e in events[:-1]][0] == "plan" and events[-2]["node"] == "respond"
        assert events[-1] == {"type": "status", "status": "done"}


def test_info_endpoint(env):
    with env.client() as c:
        assert c.get("/api/v1/info").status_code == 401
        info = c.get("/api/v1/info", headers=R).json()
        assert info == {"repo": "o/r", "sandbox_repo": "me/sandbox", "role": "reader"}


def test_chat_page_is_served(env):
    with env.client() as c:
        r = c.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        html = r.text
        for marker in ('id="question"', 'id="approve"', 'id="reject"', 'id="api-key"', "/events", "X-API-Key"):
            assert marker in html, marker
        assert "new EventSource" not in html  # EventSource can't send the X-API-Key header


def test_make_graph_factory_wires_a_working_graph(env, _test_database):
    from copilot.service.wiring import make_graph_factory

    factory = make_graph_factory(env.settings.model_copy(update={"database_url": _test_database}),
                                 reranker=FakeReranker(), tools=FakeTools(), write_tools=env.writes,
                                 chat=env.chat, embedder=FakeEmbedder())
    with factory() as graph:
        out = graph.invoke(new_turn_input("q"), {"configurable": {"thread_id": uuid.uuid4().hex[:8]}})
    assert out["answer"]["sources"]


def test_periodic_recovery_picks_up_runs_whose_lease_expires_after_startup(env):
    # A crash followed by a quick restart: at startup the dead run's lease hasn't expired yet,
    # so only a periodic re-check can resume it.
    import time

    with env.client(recover_interval_s=0.05) as c:
        t = uuid.uuid4().hex
        env.graph().invoke(new_turn_input("q"), {"configurable": {"thread_id": t}}, interrupt_after=["retrieve"])
        env.db.execute("insert into thread_metadata (thread_id, status, lease_until)"
                       " values (%s, 'running', now() + interval '200 milliseconds')", (t,))
        env.db.commit()
        deadline = time.time() + 5
        while time.time() < deadline and c.get(f"/api/v1/threads/{t}", headers=M).json()["status"] != "done":
            time.sleep(0.05)
        assert c.get(f"/api/v1/threads/{t}", headers=M).json()["status"] == "done"


def test_lease_and_recovery_settings_defaults(env):
    assert (env.settings.run_lease_s, env.settings.recover_interval_s) == (60.0, 15.0)


def test_startup_recovers_crashed_threads(env):
    t = uuid.uuid4().hex[:8]
    env.graph().invoke(new_turn_input("q"), {"configurable": {"thread_id": t}}, interrupt_after=["retrieve"])
    env.db.execute("insert into thread_metadata (thread_id, status, lease_until)"
                   " values (%s, 'running', now() - interval '1 second')", (t,))
    env.db.commit()
    with env.client() as c:  # entering the client runs the app's startup (lifespan)
        assert c.get(f"/api/v1/threads/{t}", headers=M).json()["status"] == "done"


def test_event_stream_reuses_one_db_connection(env, monkeypatch):
    # Review #6: each SSE poll opened a new connection; a few tabs could exhaust Postgres.
    from copilot.service.runner import ThreadRunner as TR

    seen = []
    real = TR.status

    def spy(self, thread_id, conn=None):
        seen.append(id(conn) if conn is not None else None)
        return real(self, thread_id, conn)

    monkeypatch.setattr(TR, "status", spy)
    with env.client() as c:
        t = new_thread(c)
        c.post(f"/api/v1/threads/{t}/messages", json={"question": "q"}, headers=R)
        with c.stream("GET", f"/api/v1/threads/{t}/events", headers=R) as r:
            list(r.iter_lines())
    assert seen and None not in seen and len(set(seen)) == 1



# ---------------------------------------------------------------------------------------------
# Review fixes: thread ownership, full ids, framing, constant-time keys, empty edits
# ---------------------------------------------------------------------------------------------
R2 = {"X-API-Key": "r2-key"}


def test_thread_ids_are_full_128_bit(env):
    with env.client() as c:
        assert len(new_thread(c)) == 32


def test_readers_only_see_their_own_threads(env):
    with env.client() as c:
        mine = new_thread(c)  # created with R
        for path in (f"/api/v1/threads/{mine}", f"/api/v1/threads/{mine}/events"):
            assert c.get(path, headers=R2).status_code == 404  # 404, not 403: don't reveal it exists
        assert c.post(f"/api/v1/threads/{mine}/messages", json={"question": "q"}, headers=R2).status_code == 404
        assert c.get(f"/api/v1/threads/{mine}", headers=R).status_code == 200
        assert c.get(f"/api/v1/threads/{mine}", headers=M).status_code == 200  # maintainers see all


def test_reader_cannot_use_a_maintainer_thread(env):
    with env.client() as c:
        theirs = c.post("/api/v1/threads", headers=M).json()["thread_id"]
        assert c.get(f"/api/v1/threads/{theirs}", headers=R).status_code == 404


@pytest.mark.parametrize("method,path,body", [
    ("get", "", None), ("get", "/events", None),
    ("post", "/messages", {"question": "q"}), ("post", "/approve", {"approved": True}),
])
def test_unknown_thread_is_404(env, method, path, body):
    with env.client() as c:
        tid_ = uuid.uuid4().hex
        r = getattr(c, method)(f"/api/v1/threads/{tid_}{path}", headers=M, **({"json": body} if body else {}))
        assert r.status_code == 404


def test_chat_page_cannot_be_framed_and_has_a_csp(env):
    with env.client() as c:
        r = c.get("/")
        assert r.headers["x-frame-options"] == "DENY"
        csp = r.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp and "default-src 'self'" in csp


def test_empty_edited_body_rejected_by_the_api(env):
    with env.client() as c:
        t = _awaiting(env, c)
        assert c.post(f"/api/v1/threads/{t}/approve", json={"approved": True, "body": ""}, headers=M).status_code == 422
        assert env.writes.calls == []


def test_api_key_check_is_constant_time(env, monkeypatch):
    import hmac

    from copilot.service import api as api_module
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(api_module.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    with env.client() as c:
        assert c.get("/api/v1/info", headers=R).status_code == 200
    assert calls  # compared with hmac.compare_digest, not ==/dict lookup


def test_failed_run_can_be_retried_over_http(env):
    env.chat = FakeChat({"plan": [plan()], "answer": [RuntimeError("model down"), cite_first]})
    with env.client() as c:
        t = new_thread(c)
        c.post(f"/api/v1/threads/{t}/messages", json={"question": "q"}, headers=R)
        assert c.get(f"/api/v1/threads/{t}", headers=R).json()["status"] == "failed"
        assert c.post(f"/api/v1/threads/{t}/retry", headers=R).status_code == 202
        assert c.get(f"/api/v1/threads/{t}", headers=R).json()["status"] == "done"
        assert c.post(f"/api/v1/threads/{t}/retry", headers=R).status_code == 409  # nothing to retry
