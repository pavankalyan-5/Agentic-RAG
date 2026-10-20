"""OpenTelemetry tracing with OpenInference conventions, exported to Arize Phoenix.

A *span* is one timed step (with attributes); spans nest into a *trace* = one agent turn.
All spans are manual (span() below), following OpenInference attribute conventions:
- graph nodes: wrapped in build_graph. Auto-instrumentation (OpenInference's LangChain instrumentor)
  was tried first, but LangGraph runs nodes on worker threads where the instrumentor's node span
  isn't the current context, so retrieval spans attached to the root instead of their node.
- LLM calls: traced in the gateway rather than by auto-instrumenting the OpenAI SDK, because the
  gateway also sees cache hits (which never reach OpenAI) and cost; both would double-count tokens.
- retrieval, reranking, tool calls: wrapped where they happen.

Tracing is off unless COPILOT_PHOENIX_ENDPOINT is set. When it's off, span() returns OpenTelemetry's
no-op span, so instrumented code costs ~nothing. Export failures are logged by the SDK, never raised.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from copilot.config import Settings

log = logging.getLogger("copilot.tracing")

_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None

MAX_DOCS = 10
MAX_CONTENT = 300


def setup_tracing(settings: Settings, *, exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Start tracing (idempotent). Pass an exporter in tests; otherwise OTLP to settings.phoenix_endpoint."""
    global _provider, _tracer
    if _provider is not None:
        return _provider
    if exporter is None and not settings.phoenix_endpoint:
        return None

    provider = TracerProvider(resource=Resource.create({
        "service.name": "dev-docs-copilot",
        "openinference.project.name": settings.phoenix_project,  # Phoenix groups traces by project
    }))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        # Batched in a background thread: a slow or dead Phoenix never slows a turn down.
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.phoenix_endpoint)))

    _provider, _tracer = provider, provider.get_tracer("copilot")
    log.info("tracing on → %s", settings.phoenix_endpoint or type(exporter).__name__)
    return provider


def reset_tracing() -> None:
    """Undo setup_tracing (tests)."""
    global _provider, _tracer
    if _provider is not None:
        _provider.shutdown()
    _provider = _tracer = None


def shutdown_tracing() -> None:
    """Flush buffered spans before the process exits."""
    if _provider is not None:
        _provider.force_flush()


@contextmanager
def span(name: str, kind: str, **attributes) -> Iterator[trace.Span]:
    """A manual span. kind is an OpenInference span kind: CHAIN, RETRIEVER, RERANKER, TOOL, LLM."""
    tracer = _tracer or trace.get_tracer("copilot")  # no provider -> OpenTelemetry's no-op tracer
    with tracer.start_as_current_span(name) as current:
        if current.is_recording():
            current.set_attribute("openinference.span.kind", kind)
            set_attributes(current, **attributes)
        yield current


def set_attributes(target: trace.Span, **attributes) -> None:
    if not target.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if not isinstance(value, (str, bool, int, float)):
            value = json.dumps(value, default=str)
        target.set_attribute(key.replace("__", "."), value)  # llm__model_name -> llm.model_name


def record_documents(target: trace.Span, docs: list, prefix: str = "retrieval.documents") -> None:
    """OpenInference document attributes, so Phoenix shows what retrieval returned."""
    if not target.is_recording():
        return
    for i, d in enumerate(docs[:MAX_DOCS]):
        target.set_attribute(f"{prefix}.{i}.document.id", str(d.chunk_id))
        target.set_attribute(f"{prefix}.{i}.document.score", float(d.score))
        target.set_attribute(f"{prefix}.{i}.document.content", d.content[:MAX_CONTENT])
        target.set_attribute(f"{prefix}.{i}.document.metadata",
                             json.dumps({"type": d.doc_type, "number": d.number, "title": d.title, "url": d.url}))


def traced_node(name: str, fn, *, wants_config: bool = False):
    """Wrap a graph node in a CHAIN span named after the node. The span is opened inside the node's
    own execution, so everything the node does (search, rerank, LLM, tools) nests under it."""
    from langchain_core.runnables import RunnableConfig

    if wants_config:
        def node(state: dict, config: RunnableConfig):  # annotation lets LangGraph inject config
            with span(name, "CHAIN"):
                return fn(state, config)
    else:
        def node(state: dict):
            with span(name, "CHAIN"):
                return fn(state)
    node.__name__ = name
    return node
