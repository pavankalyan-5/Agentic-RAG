"""Command-line interface: copilot init-db | sync | search | spend."""

import uuid
from pathlib import Path
from typing import Annotated

import psycopg
import typer

from copilot.agent.graph import build_graph, open_checkpointer, run_turn
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
    question: str,
    thread: Annotated[str | None, typer.Option(help="continue an earlier conversation")] = None,
) -> None:
    """Ask the agent a question; answers cite their sources."""
    settings = get_settings()
    thread = thread or uuid.uuid4().hex[:8]
    with connect(settings.database_url) as conn, open_checkpointer(settings.database_url) as checkpointer:
        before = spent_usd(conn)
        deps = AgentDeps(conn=conn, llm=build_chat(conn, settings), embedder=build_embedder(conn, settings),
                         reranker=build_reranker(settings), settings=settings)
        try:
            answer = run_turn(build_graph(deps, checkpointer), thread, question)
        except BudgetExceeded as exc:
            typer.echo(f"Stopped: budget ceiling reached ({exc}). thread: {thread}", err=True)
            raise typer.Exit(1)
        except (LLMOutputError, ProviderError) as exc:
            typer.echo(f"Sorry, I couldn't get an answer from the model ({type(exc).__name__}: {exc}). "
                       f"Try again in a moment. thread: {thread}", err=True)
            raise typer.Exit(1)
        typer.echo(f"\n{answer['text']}\n")
        if answer["sources"]:
            typer.echo("Sources:")
            for s in answer["sources"]:
                label = f"[{s['doc_type']} #{s['number']}]" if s["number"] is not None else f"[{s['doc_type']}]"
                typer.echo(f"  [{s['chunk_id']}] {label} {s['title']} — {s['url']}")
        after = spent_usd(conn)
        typer.echo(f"\nthread: {thread} · this question: ${after - before:.5f} · "
                   f"total: ${after:.4f} of ${settings.budget_usd:.2f}")


@app.command()
def spend() -> None:
    """Show total OpenAI spend against the ceiling."""
    settings = get_settings()
    with connect(settings.database_url) as conn:
        _print_spend(conn, settings)


def _print_spend(conn: psycopg.Connection, settings: Settings) -> None:
    typer.echo(f"Spent ${spent_usd(conn):.4f} of ${settings.budget_usd:.2f} ceiling")
