"""Graph nodes. Each takes the current state (+ injected deps) and returns a partial state update."""

import re
from dataclasses import asdict, dataclass

import psycopg

from copilot.agent.prompts import (
    ANSWER_SYSTEM, CITATION, NO_RESULTS_ANSWER, NO_SEARCH_ANSWER, PLANNER_SYSTEM, render_context, render_history,
)
from copilot.agent.schemas import AnswerDraft, Plan
from copilot.config import Settings
from copilot.llm.gateway import ChatModel, Embedder
from copilot.retrieval.agentic import SubQuery, retrieve_for_subqueries
from copilot.retrieval.rerank import Reranker

MAX_SUBQUERIES = 3
MAX_CORRECTIONS = 1
# Fenced blocks and inline code spans: brackets there are code (arr[0]), never citations.
_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")
UNGROUNDED_WARNING = (
    "\n\n⚠️ I couldn't link this answer to sources in the indexed docs, issues, or PRs — "
    "verify it before relying on it."
)


@dataclass
class AgentDeps:
    conn: psycopg.Connection | None
    llm: ChatModel
    embedder: Embedder
    reranker: Reranker
    settings: Settings


def plan_node(state: dict, deps: AgentDeps) -> dict:
    question = state["question"]
    history = render_history(state.get("messages", []))
    user = f"Conversation so far:\n{history}\n\nLatest question: {question}" if history else question
    plan = deps.llm.chat(
        [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}],
        model=deps.settings.planner_model, schema=Plan, operation="plan",
    )
    if not plan.needs_search:
        return {"plan": []}
    sub_queries = plan.sub_queries[:MAX_SUBQUERIES] or [SubQuery(query=question)]
    return {"plan": [sq.model_dump() for sq in sub_queries]}


def retrieve_node(state: dict, deps: AgentDeps) -> dict:
    sub_queries = [SubQuery.model_validate(p) for p in state.get("plan", [])]
    if not sub_queries:
        return {"retrieved_docs": []}
    hits = retrieve_for_subqueries(
        deps.conn, deps.embedder, deps.reranker, sub_queries,
        keep=deps.settings.rerank_keep, min_score=deps.settings.rerank_min_score,
    )
    return {"retrieved_docs": [asdict(h) for h in hits]}


def synthesize_node(state: dict, deps: AgentDeps) -> dict:
    docs = state.get("retrieved_docs", [])
    if not docs:
        # No evidence -> no LLM call: a fixed reply can't hallucinate and costs nothing.
        text = NO_SEARCH_ANSWER if not state.get("plan") else NO_RESULTS_ANSWER
        return {"draft": {"answer": text, "cited_chunk_ids": [], "insufficient_context": bool(state.get("plan"))}}

    history = render_history(state.get("messages", []))
    user = f"<context>\n{render_context(docs)}\n</context>\n\n"
    if history:
        user += f"Conversation so far:\n{history}\n\n"
    user += f"Question: {state['question']}"
    messages = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]
    if state.get("error_context"):
        messages.append({"role": "user", "content": f"Correction needed: {state['error_context']}"})

    draft = deps.llm.chat(messages, model=deps.settings.answer_model, schema=AnswerDraft, operation="answer")
    return {"draft": draft.model_dump()}


def verify_node(state: dict, deps: AgentDeps) -> dict:
    """Every citation must point at a retrieved chunk, and a non-"insufficient" answer must cite
    something. One self-correction; then invalid markers are stripped and an ungrounded answer is
    flagged with a visible warning instead of being shown as if it had sources."""
    draft = state["draft"]
    docs = {d["chunk_id"]: d for d in state.get("retrieved_docs", [])}
    text = draft["answer"]

    if draft["insufficient_context"]:
        return _done(_strip_citations(text, keep=set()), [], docs, insufficient=True)
    if not docs:
        # Nothing was retrieved, so synthesize returned a fixed reply (e.g. small talk): nothing to cite.
        return _done(text, [], docs, insufficient=False)

    cited = _citations_in_prose(text) + list(draft["cited_chunk_ids"])
    invalid = sorted({c for c in cited if c not in docs})
    valid = [c for c in cited if c in docs]

    problems = []
    if invalid:
        problems.append(f"You cited chunk ids {invalid} that are not in the context.")
    if not valid:
        problems.append("Your answer has no citations to the context.")
    if problems and state.get("correction_attempts", 0) < MAX_CORRECTIONS:
        return {
            "next_step": "retry",
            "error_context": " ".join(problems) + f" Cite only these ids: {sorted(docs)}, right after each "
            "fact they support. If the context doesn't answer the question, set insufficient_context.",
            "correction_attempts": state.get("correction_attempts", 0) + 1,
        }
    if invalid:
        text = _strip_citations(text, keep=set(docs))
    if not valid:
        text += UNGROUNDED_WARNING
    return _done(text, cited, docs, insufficient=False)


def _prose_segments(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, segment) pairs; re.split with a capture group alternates prose/code."""
    return [(i % 2 == 1, part) for i, part in enumerate(_CODE.split(text))]


def _citations_in_prose(text: str) -> list[int]:
    return [
        int(n) for is_code, seg in _prose_segments(text) if not is_code
        for group in CITATION.findall(seg) for n in group.split(",")
    ]


def _strip_citations(text: str, keep: set[int]) -> str:
    """Drop citation ids not in `keep` (only in prose), then tidy the gaps left behind."""

    def fix(match: re.Match) -> str:
        ids = [n.strip() for n in match.group(1).split(",") if int(n) in keep]
        return f"[{', '.join(ids)}]" if ids else ""

    out = []
    for is_code, seg in _prose_segments(text):
        if not is_code:
            seg = CITATION.sub(fix, seg)
            seg = re.sub(r"[ \t]+([.,;:!?])", r"\1", seg)  # "Fact ." -> "Fact."
            seg = re.sub(r"[ \t]{2,}", " ", seg)
        out.append(seg)
    return "".join(out).strip()


def _done(text: str, cited: list[int], docs: dict[int, dict], insufficient: bool) -> dict:
    order = list(dict.fromkeys(c for c in cited if c in docs))  # unique, first-citation order
    sources = [
        {k: docs[c][k] for k in ("chunk_id", "title", "url", "doc_type", "number")} for c in order
    ]
    return {
        "next_step": "done",
        "error_context": None,
        "answer": {"text": text, "sources": sources, "insufficient_context": insufficient},
    }
