"""Run real questions through the agent and record everything needed to judge it.

For each question: the planner's sub-queries, ALL hybrid-search candidates per sub-query with
their reranker scores (kept and dropped — needed to choose a cutoff), the answer, sources,
cost, and wall time. Writes JSON (raw) + Markdown (readable) to docs/results/.

Usage: uv run python scripts/agent_report.py
Costs real money (~$0.002 per question; repeats are cached at $0).
"""

import json
import time
from datetime import date
from pathlib import Path

from copilot.agent.graph import build_graph, open_checkpointer, run_turn
from copilot.agent.nodes import AgentDeps
from copilot.config import get_settings
from copilot.db.connection import connect
from copilot.llm.gateway import LLMGateway, spent_usd
from copilot.retrieval.agentic import SubQuery
from copilot.retrieval.models import SearchFilters
from copilot.retrieval.rerank import CrossEncoderReranker
from copilot.retrieval.search import search

RUN = date.today().isoformat() + "-" + time.strftime("%H%M")
# (thread label, question). Same label = same conversation.
QUESTIONS = [
    ("lifespan", "How do I run code when the application starts up and shuts down?"),
    ("uploads", "How can I limit the size of uploaded files in multipart forms?"),
    ("middleware", "What changed recently in BaseHTTPMiddleware?"),
    ("middleware", "Has that change been merged, and who made it?"),
    ("jwt", "How do I add JWT authentication to my Starlette app?"),
    ("lifespan-repeat", "How do I run code when the application starts up and shuts down?"),
    ("smalltalk", "thanks!"),
]


def main() -> None:
    settings = get_settings()
    out_dir = Path("docs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = {label: f"report-{RUN}-{label}-{int(time.time())}" for label, _ in QUESTIONS}
    results = []

    with connect(settings.database_url) as conn, open_checkpointer(settings.database_url) as cp:
        gateway = LLMGateway(conn, settings)
        reranker = CrossEncoderReranker(settings.reranker_model)
        reranker.score("warm up", ["load the model before timing"])
        graph = build_graph(AgentDeps(conn, gateway, gateway, reranker, settings), cp)

        for label, question in QUESTIONS:
            before, t0 = spent_usd(conn), time.perf_counter()
            answer = run_turn(graph, threads[label], question)
            seconds, cost = time.perf_counter() - t0, spent_usd(conn) - before
            state = graph.get_state({"configurable": {"thread_id": threads[label]}}).values

            sub_queries = []
            for p in state["plan"]:
                sq = SubQuery.model_validate(p)
                filters = SearchFilters(doc_types=tuple(sq.doc_types) if sq.doc_types else None, state=sq.state)
                candidates = search(conn, gateway, sq.query, mode="hybrid", k=10, filters=filters)
                scores = reranker.score(sq.query, [c.content for c in candidates])
                # "kept" = what THIS sub-query contributed: its top rerank_keep at/above the cutoff.
                passing = sorted((s, c.chunk_id) for c, s in zip(candidates, scores)
                                 if settings.rerank_min_score is None or s >= settings.rerank_min_score)
                kept_ids = {cid for _, cid in sorted(passing, reverse=True)[: settings.rerank_keep]}
                sub_queries.append({
                    "query": sq.query, "doc_types": sq.doc_types, "state": sq.state,
                    "candidates": sorted(
                        [{"chunk_id": c.chunk_id, "score": round(s, 2), "kept": c.chunk_id in kept_ids,
                          "doc_type": c.doc_type, "number": c.number, "title": c.title, "url": c.url,
                          "preview": " ".join(c.content.split())[:140]}
                         for c, s in zip(candidates, scores)],
                        key=lambda x: x["score"], reverse=True),
                })

            results.append({"thread": label, "question": question, "seconds": round(seconds, 2),
                            "cost_usd": round(cost, 6), "sub_queries": sub_queries, "answer": answer})
            print(f"[{label}] ${cost:.5f} {seconds:.1f}s — {question}")

    (out_dir / f"{RUN}-agent-run.json").write_text(json.dumps(results, indent=2))
    (out_dir / f"{RUN}-agent-run.md").write_text(render_markdown(results, settings))
    print(f"wrote docs/results/{RUN}-agent-run.md and .json")


def render_markdown(results: list[dict], settings) -> str:
    total = sum(r["cost_usd"] for r in results)
    lines = [
        f"# Agent run — {RUN}",
        "",
        f"Repo: `{settings.demo_repo}` · planner `{settings.planner_model}` · answer `{settings.answer_model}` · "
        f"reranker `{settings.reranker_model}` · keep {settings.rerank_keep}/sub-query · "
        f"min_score {settings.rerank_min_score}",
        "",
        f"**{len(results)} questions · total ${total:.5f}**",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [f"## Q{i}. {r['question']}", "",
                  f"Thread `{r['thread']}` · ${r['cost_usd']:.5f} · {r['seconds']} s", ""]
        if not r["sub_queries"]:
            lines += ["_Planner decided no search was needed._", ""]
        for sq in r["sub_queries"]:
            filt = ", ".join(x for x in [f"types={sq['doc_types']}" if sq["doc_types"] else "",
                                         f"state={sq['state']}" if sq["state"] else ""] if x)
            lines += [f"**Sub-query:** `{sq['query']}`" + (f" ({filt})" if filt else ""), "",
                      "| score | kept | source | preview |", "|---:|:---:|---|---|"]
            for c in sq["candidates"]:
                ref = f"{c['doc_type']} #{c['number']}" if c["number"] else f"{c['doc_type']}: {c['title']}"
                preview = c["preview"].replace("|", "\\|")
                lines.append(f"| {c['score']} | {'✅' if c['kept'] else ''} | [{ref}]({c['url']}) | {preview} |")
            lines.append("")
        a = r["answer"]
        lines += ["**Answer:**", "", "> " + a["text"].replace("\n", "\n> "), ""]
        if a["sources"]:
            lines.append("**Sources:** " + " · ".join(
                f"[{s['chunk_id']}] [{s['doc_type']}{' #' + str(s['number']) if s['number'] else ': ' + s['title']}]({s['url']})"
                for s in a["sources"]))
        else:
            lines.append("**Sources:** none" + (" (insufficient context)" if a["insufficient_context"] else ""))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
