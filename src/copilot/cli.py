"""Command-line interface: copilot init-db | sync | search | ask | serve | spend."""

import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated

import psycopg
import typer

from copilot.agent.graph import (
    ThreadAwaitingApproval, build_graph, is_awaiting_approval, open_checkpointer, resume_turn, run_turn,
)
from copilot.tools.mcp_client import ToolClient, ToolError, open_github_tools
from copilot.agent.nodes import AgentDeps
from copilot.config import Settings, get_settings
from copilot.db.connection import apply_schema, connect
from copilot.ingest.github_api import GitHubClient
from copilot.ingest.pipeline import sync_repo
from copilot.llm.gateway import (
    BudgetExceeded, ChatModel, Embedder, LLMGateway, LLMOutputError, ProviderError, spent_usd,
)
from copilot.retrieval.models import SearchFilters
from copilot.retrieval.rerank import CrossEncoderReranker, Reranker
from copilot.retrieval.search import search as run_search

app = typer.Typer(help="Dev Docs Copilot", no_args_is_help=True)

WORKDIR = Path("data/repos")


def build_embedder(conn: psycopg.Connection, settings: Settings) -> Embedder:
    return LLMGateway(conn, settings)


def build_chat(conn: psycopg.Connection, settings: Settings) -> ChatModel:
    return LLMGateway(conn, settings)


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings.reranker_model)


@contextmanager
def build_tools(settings: Settings, write: bool = False) -> Iterator[ToolClient | None]:
    """Open GitHub's MCP server (Docker) if configured; yield None (feature off) if unavailable."""
    try:
        client = open_github_tools(settings, write=write)
        client.__enter__()
    except ToolError as exc:
        if write or settings.github_token:  # only mention it when the user configured it
            typer.echo(f"(GitHub {'write' if write else 'live'} tools unavailable: {exc})", err=True)
        yield None
        return
    try:
        yield client
    finally:
        client.__exit__(None, None, None)


@app.command("init-db")
def init_db() -> None:
    """Create tables and indexes (safe to re-run)."""
    with connect(get_settings().database_url) as conn:
        apply_schema(conn)
    typer.echo("Schema ready.")


@app.command()
def sync(
    repo: Annotated[str | None, typer.Option(help="owner/name; defaults to COPILOT_DEMO_REPO")] = None,
    since_months: Annotated[int | None, typer.Option(help="look-back window for the first (or --full) sync")] = None,
    full: Annotated[bool, typer.Option(help="ignore the checkpoint and re-process the whole window")] = False,
) -> None:
    """Fetch docs, issues, and PRs; redact, chunk, embed, and store them."""
    settings = get_settings()
    repo = repo or settings.demo_repo
    if not repo:
        raise typer.BadParameter("pass --repo or set COPILOT_DEMO_REPO")
    with connect(settings.database_url) as conn:
        gh = GitHubClient(settings.github_token)
        try:
            results = sync_repo(conn, build_embedder(conn, settings), gh, repo,
                                since_months or settings.since_months, WORKDIR, full=full)
        except BudgetExceeded as exc:
            typer.echo(f"Stopped: budget ceiling reached ({exc}). Progress so far is saved.", err=True)
            raise typer.Exit(1)
        for source, stats in results.items():
            typer.echo(f"{source:8} seen={stats.seen} upserted={stats.upserted} "
                       f"unchanged={stats.skipped_unchanged} chunks={stats.chunks_written}")
        _print_spend(conn, settings)


@app.command()
def search(
    query: str,
    mode: Annotated[str, typer.Option(help="dense | sparse | hybrid")] = "hybrid",
    k: Annotated[int, typer.Option("-k", help="number of results")] = 5,
    doc_type: Annotated[list[str] | None, typer.Option("--type", help="doc | issue | pr (repeatable)")] = None,
    state: Annotated[str | None, typer.Option(help="open | closed | merged")] = None,
) -> None:
    """Search the index."""
    if mode not in ("dense", "sparse", "hybrid"):
        raise typer.BadParameter("mode must be dense, sparse, or hybrid")
    settings = get_settings()
    filters = SearchFilters(doc_types=tuple(doc_type) if doc_type else None, state=state)
    with connect(settings.database_url) as conn:
        hits = run_search(conn, build_embedder(conn, settings), query, mode=mode, k=k, filters=filters)
    if not hits:
        typer.echo("No results.")
        return
    for i, h in enumerate(hits, start=1):
        label = f"[{h.doc_type} #{h.number}]" if h.number is not None else f"[{h.doc_type}]"
        snippet = " ".join(h.content.split())[:200]
        typer.echo(f"{i}. {h.score:.4f}  {label}  {h.title}\n   {h.url}\n   {snippet}\n")


@app.command()
def ask(
    question: Annotated[str | None, typer.Argument(help="the question (omit with --approve/--reject)")] = None,
    thread: Annotated[str | None, typer.Option(help="continue an earlier conversation")] = None,
    approve: Annotated[bool, typer.Option("--approve", help="approve the thread's pending GitHub action")] = False,
    reject: Annotated[bool, typer.Option("--reject", help="reject the thread's pending GitHub action")] = False,
) -> None:
    """Ask the agent a question; answers cite their sources. Actions wait for --approve/--reject."""
    deciding = approve or reject
    if deciding and (approve and reject or not thread or question):
        raise typer.BadParameter("use exactly one of --approve/--reject, with --thread and no question")
    if not deciding and not question:
        raise typer.BadParameter("ask a question, or decide a pending action with --thread and --approve/--reject")
    settings = get_settings()
    thread = thread or uuid.uuid4().hex[:8]
    with ExitStack() as stack:
        conn = stack.enter_context(connect(settings.database_url))
        checkpointer = stack.enter_context(open_checkpointer(settings.database_url))
        tools = stack.enter_context(build_tools(settings))
        write_tools = stack.enter_context(build_tools(settings, write=True)) if settings.sandbox_repo else None
        before = spent_usd(conn)
        deps = AgentDeps(conn=conn, llm=build_chat(conn, settings), embedder=build_embedder(conn, settings),
                         reranker=build_reranker(settings), settings=settings, tools=tools, write_tools=write_tools)
        graph = build_graph(deps, checkpointer)
        try:
            if deciding:
                if not is_awaiting_approval(graph, thread):
                    typer.echo(f"Nothing is waiting for approval in thread {thread}.", err=True)
                    raise typer.Exit(1)
                # The CLI runs locally with the operator's own credentials, so it acts as a maintainer.
                result = resume_turn(graph, thread, {"approved": approve, "by": "cli", "role": "maintainer",
                                                     "body": None})
            else:
                result = run_turn(graph, thread, question)
        except ThreadAwaitingApproval:
            typer.echo(f"Thread {thread} has a GitHub action waiting: decide it first with "
                       f"--thread {thread} --approve (or --reject).", err=True)
            raise typer.Exit(1)
        except BudgetExceeded as exc:
            typer.echo(f"Stopped: budget ceiling reached ({exc}). thread: {thread}", err=True)
            raise typer.Exit(1)
        except (LLMOutputError, ProviderError) as exc:
            typer.echo(f"Sorry, I couldn't get an answer from the model ({type(exc).__name__}: {exc}). "
                       f"Try again in a moment. thread: {thread}", err=True)
            raise typer.Exit(1)
        _print_result(result)
        after = spent_usd(conn)
        typer.echo(f"\nthread: {thread} · this question: ${after - before:.5f} · "
                   f"total: ${after:.4f} of ${settings.budget_usd:.2f}")
        if result["status"] == "awaiting_approval":
            typer.echo(f"Decide with: copilot ask --thread {thread} --approve   (or --reject)")


def _print_result(result: dict) -> None:
    typer.echo(f"\n{result.get('text', '')}\n")
    if result.get("sources"):
        typer.echo("Sources:")
        for s in result["sources"]:
            label = f"[{s['doc_type']} #{s['number']}]" if s["number"] is not None else f"[{s['doc_type']}]"
            typer.echo(f"  [{s['chunk_id']}] {label} {s['title']} — {s['url']}")
    pending = result.get("pending_action")
    if result["status"] == "awaiting_approval" and pending:
        target = f"issue #{pending['issue_number']}" if pending["issue_number"] else "a new issue"
        typer.echo(f"\n⏸  Awaiting approval: {pending['kind']} on {target} in {pending['repo']}")
        if pending.get("title"):
            typer.echo(f"   Title: {pending['title']}")
        typer.echo("   " + pending["body"].replace("\n", "\n   "))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="bind address (keep 127.0.0.1 unless behind auth/TLS)")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8000,
) -> None:
    """Run the API + chat page. Interrupted runs are resumed on startup."""
    import uvicorn

    from copilot.service.api import create_app
    from copilot.service.runner import ThreadRunner
    from copilot.service.wiring import make_graph_factory

    settings = get_settings()
    if not settings.api_keys:
        raise typer.BadParameter('set COPILOT_API_KEYS, e.g. \'{"<random-key>": "maintainer"}\'')
    with ExitStack() as stack:
        tools = stack.enter_context(build_tools(settings))
        write_tools = stack.enter_context(build_tools(settings, write=True)) if settings.sandbox_repo else None
        typer.echo(f"live GitHub lookups: {'on' if tools else 'off'} · "
                   f"actions: {'on → ' + settings.sandbox_repo if write_tools else 'off'}")
        factory = make_graph_factory(settings, reranker=build_reranker(settings), tools=tools,
                                     write_tools=write_tools)
        runner = ThreadRunner(factory, settings.database_url, lease_s=settings.run_lease_s)
        # Registered AFTER the MCP clients, so it runs FIRST on exit: in-flight runs (maybe mid-post)
        # finish before the tools they use are closed.
        stack.callback(runner.close)
        uvicorn.run(create_app(runner, settings), host=host, port=port)


@app.command()
def spend() -> None:
    """Show total OpenAI spend against the ceiling."""
    settings = get_settings()
    with connect(settings.database_url) as conn:
        _print_spend(conn, settings)


def _print_spend(conn: psycopg.Connection, settings: Settings) -> None:
    typer.echo(f"Spent ${spent_usd(conn):.4f} of ${settings.budget_usd:.2f} ceiling")
