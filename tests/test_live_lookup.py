import re
import uuid

import pytest

from copilot.agent.graph import build_graph, open_checkpointer, run_turn
from copilot.agent.nodes import LIVE_ID_BASE, AgentDeps, live_lookup_node, plan_node
from copilot.agent.schemas import AnswerDraft, LiveLookup, Plan
from copilot.config import Settings
from copilot.retrieval.agentic import SubQuery
from fakes import FakeChat, FakeEmbedder, FakeReranker, FakeTools


def settings() -> Settings:
    return Settings(_env_file=None, demo_repo="Kludex/starlette")


def deps(tools=None, chat=None, conn=None) -> AgentDeps:
    return AgentDeps(conn=conn, llm=chat or FakeChat(), embedder=FakeEmbedder(), reranker=FakeReranker(),
                     settings=settings(), tools=tools)


def lookups(*items) -> list[dict]:
    return [{"kind": k, "number": n} for k, n in items]


# ---------- node ----------

def test_planner_live_lookups_go_to_state():
    plan = Plan(needs_search=True, sub_queries=[SubQuery(query="trailers")],
                live_lookups=[LiveLookup(kind="pull_request", number=3564)])
    out = plan_node({"question": "Is #3564 merged?", "messages": []}, deps(chat=FakeChat({"plan": [plan]})))
    assert out["plan_live"] == [{"kind": "pull_request", "number": 3564}]


def test_live_lookup_fetches_pr_and_renders_doc():
    tools = FakeTools()
    out = live_lookup_node({"plan_live": lookups(("pull_request", 3563))}, deps(tools))
    name, args = tools.calls[0]
    assert name == "pull_request_read"
    assert args == {"method": "get", "owner": "Kludex", "repo": "starlette", "pullNumber": 3563}
    doc = out["live_docs"][0]
    assert doc["chunk_id"] == LIVE_ID_BASE and doc["doc_type"] == "live_pr" and doc["number"] == 3563
    assert (doc["state"], doc["author"]) == ("closed", "Kludex")
    assert doc["url"] == "https://github.com/Kludex/starlette/pull/3563"
    assert "not merged" in doc["content"] and "fetched live" in doc["content"]
    assert out["live_error"] is None


def test_merged_pr_state_is_merged():
    out = live_lookup_node({"plan_live": lookups(("pull_request", 3628))}, deps(FakeTools()))
    assert out["live_docs"][0]["state"] == "merged"


def test_issue_lookup_uses_issue_read():
    tools = FakeTools()
    out = live_lookup_node({"plan_live": lookups(("issue", 5))}, deps(tools))
    assert tools.calls[0][0] == "issue_read" and tools.calls[0][1]["issue_number"] == 5
    assert out["live_docs"][0]["doc_type"] == "live_issue" and out["live_docs"][0]["state"] == "open"


def test_live_lookup_capped_at_three():
    tools = FakeTools()
    live_lookup_node({"plan_live": lookups(*[("issue", n) for n in range(1, 6)])}, deps(tools))
    assert len(tools.calls) == 3


def test_tool_failure_degrades_gracefully():
    out = live_lookup_node({"plan_live": lookups(("pull_request", 1))}, deps(FakeTools(fail=True)))
    assert out["live_docs"] == [] and "unavailable" in out["live_error"]


def test_one_missing_item_does_not_drop_the_others():
    out = live_lookup_node({"plan_live": lookups(("issue", 404), ("issue", 7))}, deps(FakeTools(missing={404})))
    assert [d["number"] for d in out["live_docs"]] == [7] and "#404" in out["live_error"]


def test_no_tools_configured_degrades_gracefully():
    out = live_lookup_node({"plan_live": lookups(("issue", 1))}, deps(tools=None))
    assert out["live_docs"] == [] and "unavailable" in out["live_error"]


def test_nothing_requested_does_nothing():
    tools = FakeTools()
    assert live_lookup_node({"plan_live": []}, deps(tools)) == {"live_docs": [], "live_error": None}
    assert tools.calls == []


# ---------- graph ----------

@pytest.fixture
def make_graph(seeded_db, _test_database):
    with open_checkpointer(_test_database) as cp:
        def _make(chat, tools):
            d = AgentDeps(conn=seeded_db, llm=chat, embedder=FakeEmbedder(), reranker=FakeReranker(),
                          settings=settings(), tools=tools)
            return build_graph(d, cp, retry_initial_interval=0.01)
        yield _make


def live_plan(number: int, needs_search: bool = False) -> Plan:
    return Plan(needs_search=needs_search, sub_queries=[],
                live_lookups=[LiveLookup(kind="pull_request", number=number)])


def cite_live(messages) -> AnswerDraft:
    ids = [int(i) for i in re.findall(r'<chunk id="(\d+)"', messages[1]["content"])]
    live = next(i for i in ids if i >= LIVE_ID_BASE)
    return AnswerDraft(answer=f"It was closed without merging [{live}].", cited_chunk_ids=[live],
                       insufficient_context=False)


def test_live_docs_are_citable_end_to_end(make_graph):
    graph = make_graph(FakeChat({"plan": [live_plan(3563)], "answer": [cite_live]}), FakeTools())
    result = run_turn(graph, uuid.uuid4().hex[:8], "Is PR #3563 merged?")
    source = result["sources"][0]
    assert source["chunk_id"] == LIVE_ID_BASE and source["url"].endswith("/pull/3563")


def test_live_only_question_is_answered_not_refused(make_graph):
    # needs_search=False + a live lookup must not fall into the "Happy to help!" fixed reply.
    graph = make_graph(FakeChat({"plan": [live_plan(3563)], "answer": [cite_live]}), FakeTools())
    result = run_turn(graph, uuid.uuid4().hex[:8], "Is PR #3563 merged?")
    assert "closed without merging" in result["text"]


def test_unavailable_note_appended_to_answer(make_graph):
    chat = FakeChat({"plan": [live_plan(3563, needs_search=True)], "answer": []})
    chat.responses["plan"][0].sub_queries = [SubQuery(query="ERR_TIMEOUT client")]
    chat.responses["answer"] = [lambda m: AnswerDraft(
        answer=f"Timeouts [{re.search(r'<chunk id=.(\d+)', m[1]['content']).group(1)}].",
        cited_chunk_ids=[int(re.search(r'<chunk id=.(\d+)', m[1]['content']).group(1))],
        insufficient_context=False)]
    result = run_turn(make_graph(chat, FakeTools(fail=True)), uuid.uuid4().hex[:8], "Is #3563 merged?")
    assert result["sources"] and "live GitHub data was unavailable for #3563" in result["text"]


class WeirdTools(FakeTools):
    """Returns non-dict / incomplete replies, like an unexpected server version might."""

    def __init__(self, reply):
        super().__init__()
        self.reply = reply

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return self.reply


@pytest.mark.parametrize("reply", ["Not Found", [], {"title": "no number"}, None])
def test_unexpected_reply_degrades_instead_of_crashing(reply):
    # Review #4: a plain-text reply raised AttributeError and failed the whole run.
    out = live_lookup_node({"plan_live": lookups(("pull_request", 5))}, deps(WeirdTools(reply)))
    assert out["live_docs"] == [] and "#5" in out["live_error"]
