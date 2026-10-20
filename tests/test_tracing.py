import re
import uuid

import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from copilot.agent.graph import build_graph, open_checkpointer, run_turn
from copilot.agent.nodes import AgentDeps, live_lookup_node
from copilot.agent.schemas import AnswerDraft, LiveLookup, Plan
from copilot.config import Settings
from copilot.llm.gateway import LLMGateway
from copilot.observability.tracing import reset_tracing, setup_tracing, span
from copilot.retrieval.agentic import SubQuery
from fakes import FakeChat, FakeEmbedder, FakeOpenAI, FakeReranker, FakeTools


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    setup_tracing(Settings(_env_file=None), exporter=exp)
    yield exp
    reset_tracing()


def by_name(exporter) -> dict:
    spans = {}
    for s in exporter.get_finished_spans():
        spans.setdefault(s.name, s)
    return spans


def cite_first(messages) -> AnswerDraft:
    first = int(re.search(r'<chunk id="(\d+)"', messages[1]["content"]).group(1))
    return AnswerDraft(answer=f"It times out [{first}].", cited_chunk_ids=[first], insufficient_context=False)


@pytest.fixture
def traced_graph(seeded_db, _test_database, exporter):
    plan = Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")],
                live_lookups=[LiveLookup(kind="pull_request", number=2)])
    deps = AgentDeps(conn=seeded_db, llm=FakeChat({"plan": [plan], "answer": [cite_first]}), embedder=FakeEmbedder(),
                     reranker=FakeReranker(), settings=Settings(_env_file=None, demo_repo="o/r"), tools=FakeTools())
    with open_checkpointer(_test_database) as cp:
        yield build_graph(deps, cp, retry_initial_interval=0.01)


def test_tracing_disabled_without_endpoint():
    assert setup_tracing(Settings(_env_file=None)) is None
    with span("anything", "CHAIN") as s:  # spans are harmless no-ops when tracing is off
        assert not s.is_recording()


def test_turn_produces_one_span_tree(traced_graph, exporter):
    thread = uuid.uuid4().hex
    run_turn(traced_graph, thread, "What does ERR_TIMEOUT mean?")
    spans = by_name(exporter)
    root = spans["agent_turn"]
    assert root.attributes["session.id"] == thread
    assert root.attributes["openinference.span.kind"] == "CHAIN"
    assert root.attributes["input.value"] == "What does ERR_TIMEOUT mean?"
    assert "It times out" in root.attributes["output.value"]
    for name in ("plan", "retrieve", "live_lookup", "synthesize", "verify", "search", "rerank"):
        assert name in spans, name  # graph nodes (auto) + our retrieval spans (manual)
    trace_ids = {s.context.trace_id for s in exporter.get_finished_spans()}
    assert trace_ids == {root.context.trace_id}  # everything belongs to the same trace


def test_search_is_nested_inside_retrieve(traced_graph, exporter):
    run_turn(traced_graph, uuid.uuid4().hex, "q")
    spans = exporter.get_finished_spans()
    ids = {s.context.span_id: s for s in spans}

    def ancestors(s):
        while s.parent is not None and s.parent.span_id in ids:
            s = ids[s.parent.span_id]
            yield s.name

    search = next(s for s in spans if s.name == "search")
    assert "retrieve" in list(ancestors(search))


def test_retriever_span_lists_documents(traced_graph, exporter):
    run_turn(traced_graph, uuid.uuid4().hex, "q")
    search = by_name(exporter)["search"]
    a = search.attributes
    assert a["openinference.span.kind"] == "RETRIEVER" and a["copilot.mode"] == "hybrid"
    assert a["retrieval.documents.0.document.id"]
    assert len(a["retrieval.documents.0.document.content"]) <= 300
    rerank = by_name(exporter)["rerank"]
    assert rerank.attributes["openinference.span.kind"] == "RERANKER"


def test_tool_span_records_mcp_calls(traced_graph, exporter):
    run_turn(traced_graph, uuid.uuid4().hex, "q")
    tool = by_name(exporter)["tool.pull_request_read"]
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["tool.name"] == "pull_request_read"


def test_failed_tool_call_marks_span_error_and_turn_still_answers(exporter):
    deps = AgentDeps(conn=None, llm=FakeChat(), embedder=FakeEmbedder(), reranker=FakeReranker(),
                     settings=Settings(_env_file=None, demo_repo="o/r"), tools=FakeTools(fail=True))
    out = live_lookup_node({"plan_live": [{"kind": "issue", "number": 1}]}, deps)
    assert out["live_error"]
    tool = by_name(exporter)["tool.issue_read"]
    assert tool.status.status_code.name == "ERROR"


class Echo(BaseModel):
    text: str


def test_gateway_spans_carry_cost_and_cache_hits(db, settings, exporter):
    gw = LLMGateway(db, settings, client=FakeOpenAI(chat_responses=[Echo(text="hi")], usage=(1000, 200)))
    msgs = [{"role": "user", "content": "x"}]
    gw.chat(msgs, model="gpt-4.1-nano", schema=Echo, operation="plan")
    gw.chat(msgs, model="gpt-4.1-nano", schema=Echo, operation="plan")
    calls = [s for s in exporter.get_finished_spans() if s.name == "llm.plan"]
    assert [c.attributes["copilot.cache_hit"] for c in calls] == [False, True]
    assert calls[0].attributes["copilot.cost_usd"] == pytest.approx((1000 * 0.10 + 200 * 0.40) / 1e6)
    assert calls[1].attributes["copilot.cost_usd"] == 0
    assert calls[0].attributes["llm.model_name"] == "gpt-4.1-nano"
    assert calls[0].attributes["llm.token_count.prompt"] == 1000


class ExplodingExporter(SpanExporter):
    def export(self, spans):
        raise RuntimeError("tracing backend is down")

    def shutdown(self):
        pass


def test_exporter_failure_never_breaks_a_turn(seeded_db, _test_database):
    setup_tracing(Settings(_env_file=None), exporter=ExplodingExporter())
    try:
        plan = Plan(needs_search=True, sub_queries=[SubQuery(query="ERR_TIMEOUT client")])
        deps = AgentDeps(conn=seeded_db, llm=FakeChat({"plan": [plan], "answer": [cite_first]}),
                         embedder=FakeEmbedder(), reranker=FakeReranker(), settings=Settings(_env_file=None))
        with open_checkpointer(_test_database) as cp:
            result = run_turn(build_graph(deps, cp), uuid.uuid4().hex, "q")
        assert result["sources"]
    finally:
        reset_tracing()


def test_setup_is_idempotent(exporter):
    assert setup_tracing(Settings(_env_file=None), exporter=exporter) is setup_tracing(Settings(_env_file=None))
