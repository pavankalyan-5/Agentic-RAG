"""Command-line interface: copilot init-db | sync | search | spend."""

from pathlib import Path
from typing import Annotated

import psycopg
import typer

from copilot.config import Settings, get_settings
from copilot.db.connection import apply_schema, connect
from copilot.ingest.github_api import GitHubClient
from copilot.ingest.pipeline import sync_repo
from copilot.llm.gateway import BudgetExceeded, Embedder, LLMGateway, spent_usd
from copilot.retrieval.models import SearchFilters
from copilot.retrieval.search import search as run_search

app = typer.Typer(help="Dev Docs Copilot", no_args_is_help=True)

WORKDIR = Path("data/repos")


def build_embedder(conn: psycopg.Connection, settings: Settings) -> Embedder:
    return LLMGateway(conn, settings)


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
def spend() -> None:
    """Show total OpenAI spend against the ceiling."""
    settings = get_settings()
    with connect(settings.database_url) as conn:
        _print_spend(conn, settings)


def _print_spend(conn: psycopg.Connection, settings: Settings) -> None:
    typer.echo(f"Spent ${spent_usd(conn):.4f} of ${settings.budget_usd:.2f} ceiling")
