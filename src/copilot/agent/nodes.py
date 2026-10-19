"""Graph nodes. Each takes the current state (+ injected deps) and returns a partial state update."""

import re
import uuid
from dataclasses import asdict, dataclass

import psycopg

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from psycopg.types.json import Jsonb

from copilot.agent.prompts import (
    ACTION_SYSTEM, ANSWER_SYSTEM, CITATION, LIVE_UNAVAILABLE_NOTE, NO_RESULTS_ANSWER, NO_SEARCH_ANSWER,
    PLANNER_SYSTEM, render_context, render_history,
)
from copilot.agent.schemas import ActionDraft, AnswerDraft, Plan
from copilot.config import Settings
from copilot.llm.gateway import ChatModel, Embedder
from copilot.retrieval.agentic import SubQuery, retrieve_for_subqueries
from copilot.retrieval.rerank import Reranker
from copilot.tools.mcp_client import ToolClient, ToolError

MAX_SUBQUERIES = 3
MAX_LIVE_LOOKUPS = 3
# Live results get ids far above real chunk ids so the same [id] citation check covers them.
LIVE_ID_BASE = 9_000_000
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
    tools: ToolClient | None = None  # read-only GitHub MCP client; None disables live lookups
    write_tools: ToolClient | None = None  # sandbox-scoped write client; None disables actions


def plan_node(state: dict, deps: AgentDeps) -> dict:
    question = state["question"]
    history = render_history(state.get("messages", []))
    user = f"Conversation so far:\n{history}\n\nLatest question: {question}" if history else question
    plan = deps.llm.chat(
        [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}],
        model=deps.settings.planner_model, schema=Plan, operation="plan",
    )
    live = [lk.model_dump() for lk in plan.live_lookups[:MAX_LIVE_LOOKUPS]]
    action = plan.requested_action.model_dump() if plan.requested_action else None
    if not plan.needs_search:
        return {"plan": [], "plan_live": live, "requested_action": action}
    sub_queries = plan.sub_queries[:MAX_SUBQUERIES] or [SubQuery(query=question)]
    return {"plan": [sq.model_dump() for sq in sub_queries], "plan_live": live, "requested_action": action}


def live_lookup_node(state: dict, deps: AgentDeps) -> dict:
    """Fetch current issue/PR state from GitHub (MCP). Failures degrade to an index-only answer + note."""
    requested = state.get("plan_live", [])[:MAX_LIVE_LOOKUPS]
    if not requested:
        return {"live_docs": [], "live_error": None}
    if deps.tools is None:
        return {"live_docs": [], "live_error": LIVE_UNAVAILABLE_NOTE.format(detail=" (not configured)")}

    # The repo always comes from settings: the model only supplies numbers.
    owner, repo = deps.settings.demo_repo.split("/", 1)
    docs, failed = [], []
    for lookup in requested:
        number, is_pr = lookup["number"], lookup["kind"] == "pull_request"
        try:
            if is_pr:
                raw = deps.tools.call("pull_request_read",
                                      {"method": "get", "owner": owner, "repo": repo, "pullNumber": number})
            else:
                raw = deps.tools.call("issue_read",
                                      {"method": "get", "owner": owner, "repo": repo, "issue_number": number})
        except ToolError:
            failed.append(f"#{number}")
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("number"), int):
            # e.g. a plain-text "Not Found" or an unexpected shape from another server version
            failed.append(f"#{number}")
            continue
        docs.append(_live_doc(LIVE_ID_BASE + len(docs), raw, is_pr))

    error = None
    if failed:
        error = LIVE_UNAVAILABLE_NOTE.format(detail=f" for {', '.join(failed)}")
    return {"live_docs": docs, "live_error": error}


def _live_doc(chunk_id: int, raw: dict, is_pr: bool) -> dict:
    merged = bool(raw.get("merged"))
    state = "merged" if merged else raw.get("state")
    status = "merged" if merged else (f"{raw.get('state')} (not merged)" if is_pr else raw.get("state"))
    author = (raw.get("user") or {}).get("login")
    labels = ", ".join(lbl.get("name", "") for lbl in raw.get("labels") or []) or "none"
    content = (
        f"Live GitHub data (fetched live just now) for {'PR' if is_pr else 'issue'} #{raw['number']}: "
        f"{raw.get('title', '')}\n"
        f"status: {status} | author: {author} | labels: {labels} | comments: {raw.get('comments')} | "
        f"updated: {raw.get('updated_at')}" + (f" | merged at: {raw.get('merged_at')}" if merged else "") +
        f"\n\n{(raw.get('body') or '')[:600]}"
    )
    return {"chunk_id": chunk_id, "document_id": chunk_id, "content": content, "score": 1.0,
            "doc_type": "live_pr" if is_pr else "live_issue", "title": raw.get("title", ""),
            "url": raw.get("html_url", ""), "number": raw["number"], "state": state, "author": author}


def retrieve_node(state: dict, deps: AgentDeps) -> dict:
    sub_queries = [SubQuery.model_validate(p) for p in state.get("plan", [])]
    if not sub_queries:
        return {"retrieved_docs": []}
    hits = retrieve_for_subqueries(
        deps.conn, deps.embedder, deps.reranker, sub_queries,
        keep=deps.settings.rerank_keep, min_score=deps.settings.rerank_min_score,
    )
    return {"retrieved_docs": [asdict(h) for h in hits]}


def _context_docs(state: dict) -> list[dict]:
    return state.get("retrieved_docs", []) + state.get("live_docs", [])


def synthesize_node(state: dict, deps: AgentDeps) -> dict:
    docs = _context_docs(state)
    if not docs:
        # No evidence -> no LLM call: a fixed reply can't hallucinate and costs nothing.
        searched = bool(state.get("plan") or state.get("plan_live"))
        text = NO_RESULTS_ANSWER if searched else NO_SEARCH_ANSWER
        return {"draft": {"answer": text, "cited_chunk_ids": [], "insufficient_context": searched}}

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
    result = _verify(state)
    if result["next_step"] == "done" and state.get("live_error"):
        result["answer"]["text"] += f"\n\n{state['live_error']}"
    return result


def _verify(state: dict) -> dict:
    draft = state["draft"]
    docs = {d["chunk_id"]: d for d in _context_docs(state)}
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


# ---------------------------------------------------------------------------------------------
# Human-in-the-loop GitHub actions: propose -> approve (interrupt) -> execute (at most once)
# ---------------------------------------------------------------------------------------------


def propose_action_node(state: dict, config: RunnableConfig, deps: AgentDeps) -> dict:
    # LangGraph injects `config` only when it's annotated RunnableConfig (a plain dict annotation fails).
    """Draft the requested comment/issue. The target repo is ALWAYS the configured sandbox."""
    requested = state["requested_action"]
    sandbox = deps.settings.sandbox_repo
    if not sandbox:
        return {"pending_action": None,
                "action_result": {"status": "unavailable", "reason": "no sandbox repo is configured"}}
    if requested["kind"] == "comment_on_issue" and requested.get("issue_number") is None:
        return {"pending_action": None,
                "action_result": {"status": "unavailable", "reason": "no issue number was given"}}

    draft = deps.llm.chat(
        [{"role": "system", "content": ACTION_SYSTEM},
         {"role": "user", "content": f"Instructions: {requested['instructions']}\n\n"
                                     f"Verified answer:\n{state['answer']['text']}"}],
        model=deps.settings.answer_model, schema=ActionDraft, operation="action",
    )
    thread_id = config["configurable"]["thread_id"]
    return {"pending_action": {
        "kind": requested["kind"], "repo": sandbox, "issue_number": requested.get("issue_number"),
        "title": draft.title, "body": draft.body,
        # Random per proposal (stable once checkpointed: approve/execute re-read it from state).
        "idempotency_key": f"{thread_id}:{uuid.uuid4().hex}",
    }}


def approve_node(state: dict) -> dict:
    """Pause the graph until a human decides. The pause is checkpointed, so it survives restarts.

    On resume LangGraph re-runs this node from the top and interrupt() returns the decision, so
    nothing with side effects may happen before the interrupt() call.
    """
    decision = interrupt({"pending_action": state["pending_action"]})
    return {"approval": decision}


def execute_action_node(state: dict, deps: AgentDeps) -> dict:
    """Post an approved action AT MOST ONCE, and reconcile outcomes a crash left uncertain.

    GitHub's comment API has no idempotency keys, so exactly-once is impossible; we never post twice.
    1. Claim the action in Postgres (INSERT ... ON CONFLICT DO NOTHING) before calling GitHub.
    2. Post with a hidden marker carrying the key.
    3. Record: executed / failed (GitHub said no) / unknown (timeout or crash: it MAY have posted).
    If the claim already exists and is stale pending/unknown, look for the marker on GitHub
    (reconciliation): found -> executed; not found -> safe to post now.
    """
    pending, decision = state["pending_action"], state.get("approval") or {}
    key, by = pending["idempotency_key"], decision.get("by")

    if not decision.get("approved"):
        _record_action(deps, pending, "rejected", None, by)
        return {"action_result": {"status": "rejected"}}

    # Tool-layer authorization: checked here even though the API checks too (defense in depth).
    body = decision["body"] if decision.get("body") is not None else pending["body"]
    refusal = None
    if decision.get("role") != "maintainer":
        refusal = "only maintainers can approve actions"
    elif pending["repo"] != deps.settings.sandbox_repo or not deps.settings.sandbox_repo:
        refusal = f"actions may only target the sandbox repo ({deps.settings.sandbox_repo or 'none configured'})"
    elif deps.write_tools is None:
        refusal = "GitHub write access is not configured"
    elif not body.strip():
        refusal = "the body is empty"
    if refusal:
        return {"action_result": {"status": "refused", "reason": refusal}}

    claimed = deps.conn.execute(
        "insert into actions (idempotency_key, thread_id, kind, target_repo, payload, status, decided_by)"
        " values (%s, %s, %s, %s, %s, 'pending', %s) on conflict (idempotency_key) do nothing returning 1",
        (key, key.split(":")[0], pending["kind"], pending["repo"], Jsonb(pending), by),
    ).fetchone()
    deps.conn.commit()
    if not claimed:
        outcome = _reconcile(deps, pending)
        if outcome is not None:
            return {"action_result": outcome}
    return {"action_result": _post_and_record(deps, pending, body)}


def _post_and_record(deps: AgentDeps, pending: dict, body: str) -> dict:
    key = pending["idempotency_key"]
    owner, repo = pending["repo"].split("/", 1)
    marked = f"{body}\n\n{_marker(key)}"  # added at post time, so an edited body keeps it
    try:
        if pending["kind"] == "comment_on_issue":
            raw = deps.write_tools.call("add_issue_comment", {
                "owner": owner, "repo": repo, "issue_number": pending["issue_number"], "body": marked})
        else:
            raw = deps.write_tools.call("issue_write", {
                "method": "create", "owner": owner, "repo": repo, "title": pending["title"], "body": marked})
    except ToolError as exc:
        result = {"status": "unknown" if exc.uncertain else "failed", "reason": str(exc)}
    except Exception as exc:  # we can't know whether GitHub acted before this happened
        result = {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        result = {"status": "executed", "url": result_url(raw)}
    _store_result(deps, key, result)
    return result


def _reconcile(deps: AgentDeps, pending: dict) -> dict | None:
    """Someone already claimed this action. Return its outcome, or None if WE should post now."""
    key = pending["idempotency_key"]
    row = deps.conn.execute(
        "select status, result, extract(epoch from now() - decided_at) as age_s from actions"
        " where idempotency_key = %s", (key,),
    ).fetchone()
    stored = {"status": row["status"], **(row["result"] or {})}
    if row["status"] not in ("pending", "unknown"):
        return stored  # executed / rejected / failed: final
    # A recent claim may belong to a worker that's mid-call right now: don't race it.
    if row["age_s"] < deps.settings.mcp_timeout_s + deps.settings.run_lease_s:
        return {"status": "pending", "reason": "another attempt is in progress"}

    if pending["kind"] != "comment_on_issue":
        result = {"status": "unknown", "reason": "can't verify whether the issue was created; check GitHub"}
        _store_result(deps, key, result)
        return result
    found = _find_marked_comment(deps, pending)
    if found is _UNVERIFIABLE:
        result = {"status": "unknown", "reason": "couldn't check GitHub for an earlier attempt; check manually"}
        _store_result(deps, key, result)
        return result
    if found is not None:
        result = {"status": "executed", "url": found}
        _store_result(deps, key, result)
        return result
    # Verified: the earlier attempt never reached GitHub. Re-claim (only one reconciler wins) and post.
    won = deps.conn.execute(
        "update actions set status = 'pending', decided_at = now() where idempotency_key = %s"
        " and status in ('pending', 'unknown') and decided_at < now() - make_interval(secs => %s) returning 1",
        (key, deps.settings.mcp_timeout_s + deps.settings.run_lease_s),
    ).fetchone()
    deps.conn.commit()
    return None if won else {"status": "pending", "reason": "another attempt is in progress"}


_UNVERIFIABLE = object()
_MAX_COMMENT_PAGES = 5


def _find_marked_comment(deps: AgentDeps, pending: dict):
    """URL of the comment carrying this action's marker, None if absent, or _UNVERIFIABLE."""
    if deps.tools is None:
        return _UNVERIFIABLE
    owner, repo = pending["repo"].split("/", 1)
    marker = _marker(pending["idempotency_key"])
    try:
        for page in range(1, _MAX_COMMENT_PAGES + 1):
            raw = deps.tools.call("issue_read", {"method": "get_comments", "owner": owner, "repo": repo,
                                                 "issue_number": pending["issue_number"], "page": page,
                                                 "perPage": 100})
            comments = list(_dicts_with_body(raw))
            for c in comments:
                if marker in c["body"]:
                    return result_url(c) or f"https://github.com/{pending['repo']}/issues/{pending['issue_number']}"
            if len(comments) < 100:
                return None
    except ToolError:
        return _UNVERIFIABLE
    return _UNVERIFIABLE  # more pages than we're willing to scan


def _dicts_with_body(raw):
    """Comment-like dicts anywhere in a response (the exact shape varies across server versions)."""
    if isinstance(raw, dict):
        if isinstance(raw.get("body"), str):
            yield raw
        for value in raw.values():
            if isinstance(value, (dict, list)):
                yield from _dicts_with_body(value)
    elif isinstance(raw, list):
        for item in raw:
            yield from _dicts_with_body(item)


def _marker(key: str) -> str:
    return f"<!-- copilot:action:{key} -->"


def _store_result(deps: AgentDeps, key: str, result: dict) -> None:
    deps.conn.execute(
        "update actions set status = %s, result = %s where idempotency_key = %s",
        (result["status"], Jsonb({k: v for k, v in result.items() if k != "status"}), key),
    )
    deps.conn.commit()


def result_url(raw) -> str | None:
    """The web link to what was created. GitHub MCP v2 returns MinimalResponse {"id", "url"}, sometimes
    nested ("Single", "comment"); REST-style payloads use html_url. Only github.com page links count:
    an api.github.com URL isn't something a person can open."""
    if not isinstance(raw, dict):
        return None
    for key in ("html_url", "url"):
        value = raw.get(key)
        if isinstance(value, str) and value.startswith("https://github.com/"):
            return value
    for value in raw.values():
        if isinstance(value, dict) and (found := result_url(value)):
            return found
    return None


def _record_action(deps: AgentDeps, pending: dict, status: str, result: dict | None, by: str | None) -> None:
    key = pending["idempotency_key"]
    deps.conn.execute(
        "insert into actions (idempotency_key, thread_id, kind, target_repo, payload, status, result, decided_by)"
        " values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict (idempotency_key) do nothing",
        (key, key.split(":")[0], pending["kind"], pending["repo"], Jsonb(pending), status,
         Jsonb(result) if result else None, by),
    )
    deps.conn.commit()


def action_outcome_text(result: dict | None) -> str:
    if not result:
        return ""
    status = result["status"]
    if status == "executed":
        return f"\n\n✅ Posted to GitHub: {result.get('url') or '(no link returned)'}"
    if status == "rejected":
        return "\n\n✋ Not posted (rejected by reviewer)."
    if status == "unknown":
        return f"\n\n⚠️ Not sure whether it was posted ({result.get('reason', '')}). Check GitHub."
    return f"\n\n⚠️ Not posted ({status}): {result.get('reason', 'unknown reason')}."
