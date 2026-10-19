import re
import uuid

import pytest

from copilot.agent.graph import build_graph, open_checkpointer, run_turn
from copilot.agent.nodes import AgentDeps
from copilot.agent.schemas import AnswerDraft, Plan
from copilot.config import Settings
from copilot.llm.gateway import BudgetExceeded, LLMOutputError
from copilot.retrieval.agentic import SubQuery
from fakes import FakeChat, FakeEmbedder, FakeReranker


@pytest.fixture
def checkpointer(_test_database):
    with open_checkpointer(_test_database) as cp:
        yield cp


def tid() -> str:
    return uuid.uuid4().hex[:8]


def plan(*queries: str) -> Plan:
    return Plan(needs_search=True, sub_queries=[SubQuery(query=q) for q in queries])


def cite_first(messages) -> AnswerDraft:
    """Answer citing the first chunk in the context (ids only exist at runtime)."""
    first = re.search(r'<chunk id="(\d+)"', messages[1]["content"]).group(1)
    return AnswerDraft(answer=f"It times out [{first}].", cited_chunk_ids=[int(first)], insufficient_context=False)


def cite(chunk_id: int) -> AnswerDraft:
    return AnswerDraft(answer=f"Claim [{chunk_id}].", cited_chunk_ids=[chunk_id], insufficient_context=False)


@pytest.fixture
def make_graph(seeded_db, checkpointer):
    def _make(chat):
        deps = AgentDeps(conn=seeded_db, llm=chat, embedder=FakeEmbedder(), reranker=FakeReranker(),
                         settings=Settings(_env_file=None))
        return build_graph(deps, checkpointer, retry_initial_interval=0.01)
    return _make


def test_end_to_end_answer_with_sources(make_graph):
    graph = make_graph(FakeChat({"plan": [plan("ERR_TIMEOUT client")], "answer": [cite_first]}))
    answer = run_turn(graph, tid(), "What does ERR_TIMEOUT mean?")
    assert answer["sources"] and answer["sources"][0]["number"] == 11
    assert answer["text"].startswith("It times out [")


def test_state_survives_new_graph_instance(make_graph):
    thread = tid()
    run_turn(make_graph(FakeChat({"plan": [plan("ERR_TIMEOUT")], "answer": [cite_first]})), thread, "q1")
    fresh = make_graph(FakeChat())  # simulates a process restart: new objects, same database
    snap = fresh.get_state({"configurable": {"thread_id": thread}})
    assert [m["role"] for m in snap.values["messages"]] == ["user", "assistant"]
    assert snap.values["answer"]["sources"]


def test_follow_up_sees_previous_turn(make_graph):
    chat = FakeChat({"plan": [plan("background tasks"), plan("background tasks PRs")],
                     "answer": [cite_first, cite_first]})
    graph, thread = make_graph(chat), tid()
    run_turn(graph, thread, "How do background tasks work?")
    run_turn(graph, thread, "What about in PRs?")
    second_planner_call = " ".join(m["content"] for m in chat.calls_for("plan")[1]["messages"])
    assert "How do background tasks work?" in second_planner_call
    snap = graph.get_state({"configurable": {"thread_id": thread}})
    assert [m["role"] for m in snap.values["messages"]] == ["user", "assistant", "user", "assistant"]


def test_per_turn_fields_reset_between_turns(make_graph):
    chat = FakeChat({"plan": [plan("ERR_TIMEOUT"), Plan(needs_search=False, sub_queries=[])],
                     "answer": [cite_first]})
    graph, thread = make_graph(chat), tid()
    run_turn(graph, thread, "What does ERR_TIMEOUT mean?")
    answer = run_turn(graph, thread, "thanks!")
    assert answer["sources"] == []  # no leftovers from turn 1


def test_malformed_planner_output_is_retried(make_graph):
    chat = FakeChat({"plan": [LLMOutputError("bad json"), plan("ERR_TIMEOUT")], "answer": [cite_first]})
    run_turn(make_graph(chat), tid(), "q")
    assert len(chat.calls_for("plan")) == 2


def test_bad_citation_triggers_one_correction(make_graph):
    chat = FakeChat({"plan": [plan("ERR_TIMEOUT")], "answer": [cite(999), cite_first]})
    answer = run_turn(make_graph(chat), tid(), "q")
    assert len(chat.calls_for("answer")) == 2
    assert "999" in chat.calls_for("answer")[1]["messages"][-1]["content"]  # error fed back
    assert answer["sources"] and all(s["chunk_id"] != 999 for s in answer["sources"])


def test_budget_exceeded_is_not_retried_and_keeps_checkpoint(make_graph):
    chat = FakeChat({"plan": [plan("ERR_TIMEOUT")], "answer": [BudgetExceeded("ceiling")]})
    graph, thread = make_graph(chat), tid()
    with pytest.raises(BudgetExceeded):
        run_turn(graph, thread, "q")
    assert len(chat.calls_for("answer")) == 1
    snap = make_graph(FakeChat()).get_state({"configurable": {"thread_id": thread}})
    assert snap.values.get("retrieved_docs")  # plan + retrieve were checkpointed before the failure
    assert snap.next == ("synthesize",)  # and the graph knows where it stopped
