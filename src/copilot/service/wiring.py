"""Builds per-run graphs for the service, sharing the expensive, thread-safe pieces."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from copilot.agent.graph import build_graph, open_checkpointer
from copilot.agent.nodes import AgentDeps
from copilot.config import Settings
from copilot.db.connection import connect
from copilot.llm.gateway import ChatModel, Embedder, LLMGateway
from copilot.retrieval.rerank import Reranker
from copilot.tools.mcp_client import ToolClient


def make_graph_factory(
    settings: Settings,
    *,
    reranker: Reranker,
    tools: ToolClient | None,
    write_tools: ToolClient | None,
    chat: ChatModel | None = None,
    embedder: Embedder | None = None,
) -> Callable[[], AbstractContextManager]:
    """Shared across runs: reranker (model loaded once) and the MCP clients (one container each;
    their blocking portal is safe to call from several threads). Per run: a DB connection, the
    gateway on it, and a checkpointer, because psycopg connections must not cross threads."""

    @contextmanager
    def factory() -> Iterator:
        with connect(settings.database_url) as conn, open_checkpointer(settings.database_url) as checkpointer:
            gateway = LLMGateway(conn, settings) if chat is None or embedder is None else None
            deps = AgentDeps(conn=conn, llm=chat or gateway, embedder=embedder or gateway, reranker=reranker,
                             settings=settings, tools=tools, write_tools=write_tools)
            yield build_graph(deps, checkpointer)

    return factory
