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
from copilot.observability.tracing import setup_tracing, shutdown_tracing
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
evals_app = typer.Typer(help="Golden dataset and evaluations", no_args_is_help=True)
app.add_typer(evals_app, name="evals")
snapshot_app = typer.Typer(help="Eval data snapshot (index + LLM cache) for CI", no_args_is_help=True)
evals_app.add_typer(snapshot_app, name="snapshot")

EVALS_DIR = Path("evals")

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
    setup_tracing(settings)  # no-op unless COPILOT_PHOENIX_ENDPOINT is set
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
    shutdown_tracing()  # flush buffered spans before the CLI exits


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
    setup_tracing(settings)
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
        stack.callback(shutdown_tracing)
        uvicorn.run(create_app(runner, settings), host=host, port=port)


def fetch_spans(settings: Settings, since_hours: float) -> list[dict]:
    """Spans from Phoenix for this project, normalized for compute_report()."""
    from datetime import datetime, timedelta, timezone

    from phoenix.client import Client

    from copilot.observability.report import rows_from_phoenix

    base_url = settings.phoenix_endpoint.split("/v1/")[0] or "http://127.0.0.1:6006"
    df = Client(base_url=base_url).spans.get_spans_dataframe(
        project_identifier=settings.phoenix_project,
        start_time=datetime.now(timezone.utc) - timedelta(hours=since_hours), limit=50_000, timeout=30,
    )
    return rows_from_phoenix(df)


@app.command("trace-report")
def trace_report(
    since_hours: Annotated[float, typer.Option(help="look back this many hours")] = 24,
    out: Annotated[Path | None, typer.Option(help="also write the markdown report here")] = None,
) -> None:
    """Latency percentiles per step, tokens per model, and cost per turn, from Phoenix traces."""
    from copilot.observability.report import compute_report, render_markdown

    md = render_markdown(compute_report(fetch_spans(get_settings(), since_hours)),
                         title=f"Trace report (last {since_hours:g} h)")
    typer.echo(md)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")


@app.command()
def spend(raw: Annotated[bool, typer.Option("--raw", help="print only the dollar amount (for scripts)")] = False) -> None:
    """Show total OpenAI spend against the ceiling."""
    settings = get_settings()
    with connect(settings.database_url) as conn:
        if raw:
            typer.echo(f"{spent_usd(conn):.6f}")
        else:
            _print_spend(conn, settings)


def _print_spend(conn: psycopg.Connection, settings: Settings) -> None:
    typer.echo(f"Spent ${spent_usd(conn):.4f} of ${settings.budget_usd:.2f} ceiling")


@evals_app.command("draft")
def evals_draft(
    docs: Annotated[int, typer.Option(help="questions drafted from docs pages")] = 30,
    history: Annotated[int, typer.Option(help="questions drafted from issues/PRs")] = 20,
    multi: Annotated[int, typer.Option(help="questions needing two sources")] = 10,
    min_chars: Annotated[int, typer.Option(help="skip documents shorter than this")] = 400,
    seed: Annotated[int, typer.Option(help="sampling seed (same seed, same sample)")] = 42,
    out: Annotated[Path, typer.Option(help="where to write drafts")] = EVALS_DIR / "golden_draft.jsonl",
) -> None:
    """Draft eval cases with an LLM for HUMAN REVIEW (plus the hand-written trick cases)."""
    from copilot.evals.dataset import load_golden, write_cases
    from copilot.evals.generate import draft_cases

    if out.name == "golden.jsonl":
        raise typer.BadParameter("golden.jsonl is human-reviewed: write drafts elsewhere, review, then promote")
    settings = get_settings()
    with connect(settings.database_url) as conn:
        before = spent_usd(conn)
        cases = draft_cases(conn, build_chat(conn, settings), repo=settings.demo_repo, seed=seed,
                            per_category={"docs": docs, "history": history, "multi_part": multi},
                            model=settings.answer_model, min_chars=min_chars)
        cost = spent_usd(conn) - before
    seeds = load_golden(EVALS_DIR / "seed_cases.jsonl") if (EVALS_DIR / "seed_cases.jsonl").exists() else []
    write_cases(out, cases + seeds)
    typer.echo(f"Wrote {len(cases)} drafted + {len(seeds)} hand-written cases to {out} (cost ${cost:.4f}).")
    typer.echo("Next: review every case (wording, answer, sources), mark ~15 in_ci, save as evals/golden.jsonl.")


@evals_app.command("check")
def evals_check(path: Annotated[Path, typer.Option(help="dataset to validate")] = EVALS_DIR / "golden.jsonl") -> None:
    """Validate a dataset against the current index."""
    from collections import Counter

    from copilot.evals.dataset import load_golden, validate_against_index

    settings = get_settings()
    cases = load_golden(path)
    with connect(settings.database_url) as conn:
        problems = validate_against_index(conn, cases, repo=settings.demo_repo)
    counts = Counter(c.category for c in cases)
    typer.echo(f"{len(cases)} cases · " + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
               + f" · in CI subset: {sum(c.in_ci for c in cases)}")
    for problem in problems:
        typer.echo(f"  ✗ {problem}")
    if problems:
        raise typer.Exit(1)
    typer.echo("OK: every expected source exists in the index.")


def _load_cases(path: Path):
    from copilot.evals.dataset import load_golden

    if not path.exists():
        raise typer.BadParameter(f"{path} not found. Review evals/golden_draft.jsonl and save it as "
                                 f"evals/golden.jsonl, or pass --path evals/golden_draft.jsonl")
    return load_golden(path)


@evals_app.command("retrieval")
def evals_retrieval(
    path: Annotated[Path, typer.Option(help="golden dataset")] = EVALS_DIR / "golden.jsonl",
    k: Annotated[int, typer.Option(help="top-k documents")] = 5,
    out: Annotated[Path | None, typer.Option(help="also write the markdown report here")] = None,
) -> None:
    """Compare search strategies on the golden set: hit rate, MRR, recall (no LLM judge)."""
    from copilot.evals.retrieval_eval import evaluate_retrieval, render_markdown

    cases = _load_cases(path)
    settings = get_settings()
    with connect(settings.database_url) as conn:
        before = spent_usd(conn)
        results = evaluate_retrieval(conn, build_embedder(conn, settings), build_reranker(settings), cases,
                                     repo=settings.demo_repo, k=k)
        cost = spent_usd(conn) - before
    md = render_markdown(results, k=k, cases=cases) + f"\n_Cost of this run: ${cost:.5f} (query embeddings)._"
    typer.echo(md)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")


@evals_app.command("run")
def evals_run(
    path: Annotated[Path, typer.Option(help="golden dataset")] = EVALS_DIR / "golden.jsonl",
    ci_only: Annotated[bool, typer.Option("--ci-only", help="only cases marked in_ci")] = False,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="skip LLM-judged metrics (rules + retrieval only)")] = False,
    baseline: Annotated[bool, typer.Option("--baseline", help="also answer without retrieval, for comparison")] = False,
    thresholds: Annotated[Path, typer.Option(help="quality gate")] = EVALS_DIR / "thresholds.json",
    out: Annotated[Path | None, typer.Option(help="also write the markdown report here")] = None,
    workers: Annotated[int, typer.Option(help="judge calls graded in parallel")] = 6,
) -> None:
    """Run the agent on the golden set, grade it, and apply the quality gate (exit 1 on failure)."""
    import json

    from copilot.evals.answer_eval import (
        EVAL_SANDBOX, gate, no_rag_baseline, render_markdown, run_cases, score_baseline, score_runs, summarize,
    )
    from copilot.evals.judge import GatewayJudge
    from copilot.evals.retrieval_eval import evaluate_retrieval

    cases = [c for c in _load_cases(path) if c.in_ci or not ci_only]
    # Eval runs: no live GitHub, no write tools, a fake sandbox name. Deterministic and side-effect free.
    settings = get_settings().model_copy(update={"sandbox_repo": EVAL_SANDBOX})
    if not settings.openai_api_key:
        _retrieval_only_run(settings, cases, json.loads(thresholds.read_text()), out)
        return
    with connect(settings.database_url) as conn, open_checkpointer(settings.database_url) as checkpointer:
        before = spent_usd(conn)
        chat, embedder, reranker = build_chat(conn, settings), build_embedder(conn, settings), build_reranker(settings)
        retrieval = evaluate_retrieval(conn, embedder, reranker, cases, repo=settings.demo_repo, k=5)["hybrid+rerank"]
        graph = build_graph(AgentDeps(conn=conn, llm=chat, embedder=embedder, reranker=reranker, settings=settings),
                            checkpointer)
        runs = run_cases(graph, conn, cases)
        judge = None if no_judge else GatewayJudge(LLMGateway(conn, settings), settings.eval_judge_model)
        scores = score_runs(runs, cases, judge, workers=workers)
        summary = summarize(scores, runs)
        if retrieval["metrics"]["n"]:
            summary["hit_rate@5"] = retrieval["metrics"]["hit_rate@5"]
            summary["mrr@5"] = retrieval["metrics"]["mrr@5"]
        base = None
        if baseline and judge is not None:
            gw = LLMGateway(conn, settings)
            base = score_baseline(no_rag_baseline(gw, cases, settings.answer_model), cases, judge, workers=workers)
        summary["cost_usd"] = spent_usd(conn) - before  # everything: agent, judge, baseline
    failures = gate(summary, json.loads(thresholds.read_text()))
    title = f"Answer evaluation ({len(cases)} cases{', CI subset' if ci_only else ''}{', no judge' if no_judge else ''})"
    md = render_markdown(summary, scores, cases, gate_failures=failures, baseline=base, title=title)
    typer.echo(md)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
    if failures:
        raise typer.Exit(1)


RETRIEVAL_ONLY_NOTE = ("> LLM-judged metrics skipped: no OpenAI key (e.g. a pull request from a fork, which "
                       "gets no secrets). Only the retrieval gate ran, on cached query embeddings.")


def _retrieval_only_run(settings: Settings, cases, thresholds: dict, out: Path | None) -> None:
    """No key: gate what needs no LLM. Query embeddings come from the snapshot's cache; a question
    that isn't cached (new or edited) needs a refreshed snapshot."""
    import openai

    from copilot.evals.answer_eval import gate
    from copilot.evals.retrieval_eval import evaluate_retrieval, render_markdown

    with connect(settings.database_url) as conn:
        try:
            results = evaluate_retrieval(conn, build_embedder(conn, settings), build_reranker(settings), cases,
                                         repo=settings.demo_repo, k=5)
        except openai.OpenAIError as e:
            typer.echo(f"A question's embedding isn't cached and there is no OpenAI key ({e}). "
                       "Refresh the snapshot: `copilot evals snapshot export`, then upload it.", err=True)
            raise typer.Exit(1)
    metrics = results["hybrid+rerank"]["metrics"]
    summary = {"hit_rate@5": metrics["hit_rate@5"], "mrr@5": metrics["mrr@5"]} if metrics["n"] else {}
    failures = gate(summary, thresholds)
    lines = [f"# Retrieval-only evaluation ({len(cases)} cases)", "", RETRIEVAL_ONLY_NOTE, "",
             f"**Quality gate: {'FAILED' if failures else 'PASSED'}**", ""]
    lines += [f"- ❌ {f}" for f in failures] + ([""] if failures else [])
    lines += ["| metric | value |", "|---|---:|"] + [f"| {k} | {v:.4f} |" for k, v in summary.items()]
    md = "\n".join(lines) + "\n\n" + render_markdown(results, k=5, cases=cases)
    typer.echo(md)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n")
    if failures:
        raise typer.Exit(1)


@snapshot_app.command("export")
def snapshot_export(path: Annotated[Path, typer.Argument(help="output .tar.gz")]) -> None:
    """Write the index + LLM cache to a snapshot file (upload it as the `eval-snapshot` release)."""
    from copilot.evals.snapshot import export_snapshot

    with connect(get_settings().database_url) as conn:
        manifest = export_snapshot(conn, path)
    tables = ", ".join(f"{t}: {spec['rows']}" for t, spec in manifest["tables"].items())
    typer.echo(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) — {tables}")


@snapshot_app.command("import")
def snapshot_import(
    path: Annotated[Path, typer.Argument(help="snapshot .tar.gz")],
    yes: Annotated[bool, typer.Option("--yes", help="don't ask before replacing the local index")] = False,
) -> None:
    """REPLACE the local index and LLM cache with a snapshot (threads and actions are kept)."""
    from copilot.evals.snapshot import SnapshotError, import_snapshot

    if not yes:
        typer.confirm("This replaces your indexed documents, chunks and LLM cache. Continue?", abort=True)
    with connect(get_settings().database_url) as conn:
        try:
            manifest = import_snapshot(conn, path)
        except SnapshotError as e:
            typer.echo(f"Snapshot not imported (nothing changed): {e}", err=True)
            raise typer.Exit(1)
    typer.echo("Imported " + ", ".join(f"{t}: {s['rows']}" for t, s in manifest["tables"].items()))
