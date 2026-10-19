# Plan 3 — Live GitHub Tools (MCP), Human-in-the-Loop Actions, API & Chat Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent can (a) look up **live** issue/PR state through GitHub's official MCP server, (b) draft a GitHub action (comment / new issue) that **pauses for human approval** and only then executes against a sandbox repo, and (c) run behind a FastAPI service with a chat page, progress streaming, and **automatic resume** of interrupted runs after a server restart.

**Architecture:** Two MCP clients over stdio to `ghcr.io/github/github-mcp-server:v2.0.2` (Docker): a **read** client (read-only mode, read-only token, tool allowlist) and a **write** client (sandbox-repo-scoped token, two write tools). A sync facade (anyio blocking portal) lets LangGraph's sync nodes call async MCP. New graph nodes: `live_lookup` (after retrieve) and `propose_action → approve (LangGraph interrupt) → execute_action` (after verify). A `ThreadRunner` runs turns in background threads, records status + lease in `thread_metadata`, streams node-level progress, and on startup resumes runs whose lease expired. FastAPI exposes it; a single static HTML page is the UI.

**Tech Stack:** mcp 2.3.0 (client + `MCPServer` for test fakes), github-mcp-server v2.0.2 (Docker image), fastapi 0.146.0, uvicorn 0.54.0, sse-starlette 3.5.0, anyio, LangGraph `interrupt`/`Command`, existing plan-1/2 stack.

**Spec:** [docs/superpowers/specs/2026-10-18-dev-docs-copilot-design.md](../specs/2026-10-18-dev-docs-copilot-design.md) — §6.3 (live tools, gated writes, sandbox repo), §6.5 (HITL, durability with background runner + lease + auto-resume), §6.9 (API + UI, SSE progress events), §6.6 authorization (minimal roles here; full guardrails in plan 5).

**Verified 2026-10-19 (probes):** read-only mode drops write tools even if listed; `pull_request_read {method:"get"}` on Kludex/starlette#3564 returns `state: closed, merged: false, user.login: Kludex` in ~1 s; MCP SDK v2 renamed `CallToolResult.isError` → **`is_error`** and `FastMCP` → **`MCPServer`** (`from mcp.server import MCPServer`).

**Out of scope:** tracing/evals (plan 4); prompt-injection classifier, query-time PII redaction, claim-level verification (plan 5); multi-user accounts.

## Global Constraints

- **The model never chooses which repo to touch.** Read lookups always target `settings.demo_repo`; write actions always target `settings.sandbox_repo`. The model only supplies issue/PR numbers and text.
- Write safety is layered: (1) read client runs with `GITHUB_READ_ONLY=1` and `GITHUB_TOOLS=issue_read,pull_request_read`; (2) write client offers only `add_issue_comment,issue_write`, uses `COPILOT_GITHUB_WRITE_TOKEN` (fine-grained, **only** the sandbox repo, Issues: read/write); (3) code checks `owner/repo == settings.sandbox_repo` before every write; (4) every write needs an explicit approval from a `maintainer` API key; (5) an action executes **at most once** (idempotency key = `thread_id:turn` stored in `actions` table).
- MCP image pinned `ghcr.io/github/github-mcp-server:v2.0.2`; Python `mcp==2.3.0`. Use `result.is_error` (v2 name).
- Live results enter the context like chunks, with ids `LIVE_ID_BASE + i` (`LIVE_ID_BASE = 9_000_000`), `doc_type` `"live_issue"` / `"live_pr"`, so the existing citation check covers them.
- MCP failures (Docker down, bad token, timeout 15 s) never fail a turn: the lookup is skipped and the answer notes that live data was unavailable.
- API auth: `X-API-Key` header; `COPILOT_API_KEYS` env = JSON `{"<key>": "reader" | "maintainer"}`. Readers can ask; only maintainers can approve.
- Budget unchanged; action drafting uses `answer_model` through the gateway (operation `"action"`).
- Tests: no Docker/network — MCP is exercised against a tiny `MCPServer` fake run as a stdio subprocess; graph/API tests use in-process fakes.

## Review Focus

1. **Approve clicked twice** (or two requests race) → the GitHub action executes exactly once; the second approval gets a clear "already decided" response. Tests in Tasks 3 and 5.
2. **Server killed while a thread awaits approval** → after restart, the pending action is still shown and approving it works (interrupt state lives in the checkpoint). Tests in Tasks 3 and 4.
3. **Server killed mid-run** → on startup the run is resumed **once** (lease prevents double-resume by two workers), and the client sees it finish. Test in Task 4.
4. **MCP unavailable** (Docker not running, bad token) → the turn still answers from the index, with a visible note; no crash, no hang beyond the 15 s timeout. Test in Task 2.
5. **A reader key tries to approve**, or an action targets a repo other than the sandbox → refused (HTTP 403 / refused in the tool layer), nothing executed. Tests in Tasks 3 and 5.

---

## File Structure

```
src/copilot/
  config.py                + sandbox_repo, github_write_token, mcp_image, mcp_timeout_s, api_keys
  db/schema.sql            + thread_metadata, actions
  tools/__init__.py
  tools/mcp_client.py      ToolClient protocol, MCPToolClient (sync facade), ToolError, open_github_tools()
  agent/schemas.py         + LiveLookup, RequestedAction, ActionDraft; Plan gains live_lookups, requested_action
  agent/state.py           + live_docs, requested_action, pending_action, action_result
  agent/nodes.py           + live_lookup_node, propose_action_node, approve_node, execute_action_node
  agent/prompts.py         + ACTION_SYSTEM; planner prompt mentions live lookups / actions
  agent/graph.py           new edges; run_turn returns TurnResult; resume_turn()
  service/__init__.py
  service/runner.py        ThreadRunner (background runs, status, lease, progress events, recovery)
  service/api.py           FastAPI app factory, routes, auth, SSE
  service/static/index.html  chat page
  cli.py                   + `serve`; `ask` prints pending action and supports --approve/--reject
tests/
  fake_mcp_server.py       MCPServer fake with issue_read, pull_request_read, add_issue_comment, issue_write
  test_mcp_client.py, test_live_lookup.py, test_actions.py, test_runner.py, test_api.py
```

---

### Task 1: MCP tool client (sync facade) + config + schema

**Files:**
- Create: `src/copilot/tools/__init__.py`, `src/copilot/tools/mcp_client.py`, `tests/fake_mcp_server.py`, `tests/test_mcp_client.py`
- Modify: `src/copilot/config.py`, `src/copilot/db/schema.sql`, `tests/conftest.py` (truncate new tables), `pyproject.toml` (`mcp==2.3.0`)

**Interfaces:**
- Settings: `sandbox_repo: str = ""`, `github_write_token: str = ""`, `mcp_image: str = "ghcr.io/github/github-mcp-server:v2.0.2"`, `mcp_timeout_s: float = 15.0`, `api_keys: dict[str, Literal["reader", "maintainer"]] = {}`.
- Schema:
  - `thread_metadata(thread_id text PK, status text NOT NULL CHECK (status IN ('idle','running','awaiting_approval','done','failed')), lease_until timestamptz, last_error text, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now())`
  - `actions(idempotency_key text PK, thread_id text NOT NULL, kind text NOT NULL, target_repo text NOT NULL, payload jsonb NOT NULL, status text NOT NULL CHECK (status IN ('executed','rejected','failed')), result jsonb, decided_by text, decided_at timestamptz NOT NULL DEFAULT now())`
- `class ToolError(Exception)` — MCP unreachable, timeout, or `result.is_error`.
- `class ToolClient(Protocol)`: `def call(self, name: str, arguments: dict) -> dict | list | str` ; `def tool_names(self) -> set[str]`.
- `class MCPToolClient` — context manager. `__init__(self, command: str, args: list[str], env: dict[str, str], timeout_s: float)`. `__enter__` starts `anyio.from_thread.start_blocking_portal()`, enters `stdio_client` + `ClientSession` through `portal.wrap_async_context_manager`, calls `initialize()` and caches tool names; `call()` runs `session.call_tool` via the portal with `anyio.fail_after(timeout_s)`, raises `ToolError` on timeout/`is_error`, returns `json.loads(text)` when the first content item is JSON text, else the text. `__exit__` closes everything (including on errors).
- `open_github_tools(settings, *, write: bool = False) -> MCPToolClient` — builds the `docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN -e GITHUB_TOOLS=... [-e GITHUB_READ_ONLY=1] <image> stdio` command; read: token `github_token`, tools `issue_read,pull_request_read`, read-only; write: token `github_write_token`, tools `add_issue_comment,issue_write`. Raises `ToolError` if the needed token is empty.
- `tests/fake_mcp_server.py`: `MCPServer("fake-github")` with `issue_read`, `pull_request_read` (returns canned JSON for numbers 1–20; `number == 404` returns an MCP error), `add_issue_comment`, `issue_write` (append each call to a JSONL file at env `FAKE_MCP_LOG`, return `{"html_url": ...}`), and `slow_tool` (sleeps 5 s).

- [ ] **Step 1: Write the failing tests** (spawn the fake with `sys.executable tests/fake_mcp_server.py`)

```python
def test_lists_tools_and_calls_one(fake_client):
    assert {"issue_read", "pull_request_read"} <= fake_client.tool_names()
    pr = fake_client.call("pull_request_read", {"method": "get", "owner": "o", "repo": "r", "pullNumber": 7})
    assert pr["number"] == 7 and pr["merged"] is False

def test_tool_error_raises(fake_client):
    with pytest.raises(ToolError):
        fake_client.call("issue_read", {"method": "get", "owner": "o", "repo": "r", "issue_number": 404})

def test_timeout_raises_tool_error(fake_server_cmd):
    with MCPToolClient(*fake_server_cmd, timeout_s=0.5) as c, pytest.raises(ToolError):
        c.call("slow_tool", {})

def test_unstartable_server_raises_tool_error():
    with pytest.raises(ToolError):
        with MCPToolClient("definitely-not-a-command", [], {}, timeout_s=2): pass

def test_open_github_tools_builds_read_only_command(settings):
    settings.github_token = "t"
    c = open_github_tools(settings)                    # not entered: no subprocess started
    assert "GITHUB_READ_ONLY=1" in c.args and "GITHUB_TOOLS=issue_read,pull_request_read" in c.args

def test_write_client_requires_write_token(settings):
    with pytest.raises(ToolError): open_github_tools(settings, write=True)
```

- [ ] **Step 2:** run → FAIL. **Step 3:** `UV_HTTP_TIMEOUT=300 uv add mcp==2.3.0`; implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 2: Live lookups in the agent

**Files:**
- Modify: `agent/schemas.py`, `agent/state.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`
- Create: `tests/test_live_lookup.py`

**Interfaces:**
- `class LiveLookup(BaseModel)`: `kind: Literal["issue", "pull_request"]`, `number: int`. `Plan` gains `live_lookups: list[LiveLookup]` (description: only when the user asks about *current* status of specific numbered items, e.g. "is #123 merged/open?"; max 3).
- `AgentDeps` gains `tools: ToolClient | None` (None = live data disabled).
- `live_lookup_node(state, deps) -> dict` → `{"live_docs": [doc dicts], "live_error": str | None}`. For each lookup (≤3): `issue_read {method:"get"}` or `pull_request_read {method:"get"}` with owner/repo from `settings.demo_repo`; render a compact text (`title`, `state`, `merged`, `merged_at`, `user.login`, `labels`, `updated_at`, `comments`, first 600 chars of `body`) into a doc dict `{"chunk_id": LIVE_ID_BASE+i, "doc_type": "live_pr"|"live_issue", "number", "title", "url": html_url, "state": "merged" if merged else state, "author": user.login, "content": ...}`. `ToolError` or `tools is None` → `live_docs: []`, `live_error: "Live GitHub data was unavailable; answered from the index only."`.
- `synthesize_node` uses `retrieved_docs + live_docs`; `verify_node` validates against both; when `live_error` is set and the plan had lookups, the note is appended to the final text.
- Graph: `retrieve → live_lookup → synthesize`.

- [ ] **Step 1: Write the failing tests** (`FakeTools` in `tests/fakes.py` implementing `ToolClient`, recording calls; raising `ToolError` when configured)

```python
def test_live_lookup_fetches_pr_and_renders_doc(): ...         # call args owner/repo from settings; doc state "closed", author "Kludex"
def test_merged_pr_state_is_merged(): ...
def test_live_lookup_capped_at_three(): ...
def test_tool_failure_degrades_gracefully():
    out = live_lookup_node({"plan_live": [{"kind": "pull_request", "number": 1}]}, deps(tools=FakeTools(fail=True)))
    assert out["live_docs"] == [] and "unavailable" in out["live_error"]
def test_no_tools_configured_degrades_gracefully(): ...
def test_live_docs_are_citable_end_to_end(make_graph): ...    # answer cites LIVE_ID_BASE; source url is the PR html_url
def test_unavailable_note_appended_to_answer(make_graph): ...
```

(Plan stores lookups in state as `plan_live: list[dict]`; `plan` keeps sub-queries.)

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 3: Human-in-the-loop GitHub actions

**Files:**
- Modify: `agent/schemas.py`, `agent/state.py`, `agent/prompts.py`, `agent/nodes.py`, `agent/graph.py`
- Create: `tests/test_actions.py`

**Interfaces:**
- `class RequestedAction(BaseModel)`: `kind: Literal["comment_on_issue", "create_issue"]`, `issue_number: int | None`, `instructions: str` (what the user asked to say). `Plan` gains `requested_action: RequestedAction | None` (only when the user explicitly asks to post/create something).
- `class ActionDraft(BaseModel)`: `title: str | None`, `body: str` — drafted by `answer_model`, operation `"action"`, from the instructions + the verified answer.
- State: `requested_action: dict | None`, `pending_action: dict | None` (`{"kind", "repo", "issue_number", "title", "body", "idempotency_key"}`), `action_result: dict | None`.
- Nodes:
  - `propose_action_node` → drafts `pending_action`; `repo` is always `settings.sandbox_repo` (empty → `action_result = {"status": "unavailable", "reason": "no sandbox repo configured"}`, no interrupt); `idempotency_key = f"{thread_id}:{len(messages)}"` (thread id from `config["configurable"]`).
  - `approve_node` → `decision = interrupt({"pending_action": pending_action})`; nothing side-effecting before `interrupt` (the node re-runs on resume). Returns `{"approval": decision}` where decision = `{"approved": bool, "by": str, "role": str, "body": str | None}` (an edited body overrides).
  - `execute_action_node(state, deps)` → if not approved: record `actions` row `rejected`, result `{"status": "rejected"}`. If approved: **refuse** unless `role == "maintainer"` and `pending_action["repo"] == settings.sandbox_repo`; then `INSERT ... ON CONFLICT (idempotency_key) DO NOTHING RETURNING` — if no row inserted, return the stored result (already executed); else call `add_issue_comment` / `issue_write {method:"create"}` via `deps.write_tools`, store result (`html_url`) and status.
  - `respond_node` appends the action outcome ("Posted: <url>" / "Not posted (rejected)") to the assistant message.
- Graph: `verify ─done→ (requested_action ? propose_action → approve → execute_action) → respond`. `AgentDeps` gains `write_tools: ToolClient | None`.
- `run_turn(...) -> TurnResult` (`{"status": "done" | "awaiting_approval", "answer": dict | None, "pending_action": dict | None}`) — detects interrupt via `graph.get_state(config).next == ("approve",)`. `resume_turn(graph, thread_id, decision: dict) -> TurnResult` → `graph.invoke(Command(resume=decision), config)`.

- [ ] **Step 1: Write the failing tests** (real `PostgresSaver` on the test DB; `FakeTools` for write)

```python
def test_requested_comment_pauses_for_approval(...): ...     # status awaiting_approval; pending repo == sandbox; nothing written
def test_pending_action_survives_restart(...):              # Review Focus 2
    run_turn(make_graph(chat), t, "comment 'thanks' on #3")
    res = resume_turn(make_graph(FakeChat()), t, {"approved": True, "by": "k", "role": "maintainer", "body": None})
    assert res["status"] == "done" and write_tools.calls[0][0] == "add_issue_comment"
def test_reject_executes_nothing(...): ...
def test_reader_cannot_execute(...): ...                     # role reader -> refused, no write call, action row status failed
def test_wrong_repo_refused_in_tool_layer(...): ...          # pending repo tampered -> refused
def test_executes_at_most_once(...):                         # Review Focus 1
    # resume twice with approval -> exactly one write call; second returns stored result
def test_edited_body_is_used(...): ...
def test_no_sandbox_configured_skips_interrupt(...): ...
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 4: Thread runner — background runs, progress, lease, auto-resume

**Files:**
- Create: `src/copilot/service/__init__.py`, `src/copilot/service/runner.py`, `tests/test_runner.py`

**Interfaces:**
- `class ThreadRunner` — `__init__(self, graph_factory: Callable[[], ContextManager[graph]], database_url: str, executor: Executor | None = None, lease_s: float = 60)`.
  - `start_turn(thread_id, question) -> None` — refuses (`ThreadBusy`) if status is `running`/`awaiting_approval`; sets `running` + `lease_until = now + lease_s`; submits the run to the executor. The run uses `graph.stream(..., stream_mode="updates")`, appending `{"node": name, "at": iso}` to an in-memory per-thread event list and renewing the lease after each node. Ends with status `done` / `awaiting_approval` / `failed` (+ `last_error`).
  - `decide(thread_id, decision) -> None` — only when status is `awaiting_approval` (else `NotAwaiting`); runs `resume_turn` in the executor the same way.
  - `snapshot(thread_id) -> dict` — status, messages, latest answer, pending_action, action_result, events (from checkpoint + metadata).
  - `events_since(thread_id, index) -> list[dict]`.
  - `recover() -> list[str]` — on startup: claim (single `UPDATE ... SET lease_until = now()+lease WHERE status='running' AND lease_until < now() RETURNING thread_id`) and resume each with `graph.invoke(None, config)` (continues from the last checkpoint). Returns resumed ids.
- Tests use `executor=InlineExecutor()` (runs synchronously) and the fake deps from plan 2.

- [ ] **Step 1: Write the failing tests**

```python
def test_turn_runs_and_records_done(...): ...                 # status done; events include plan..respond in order
def test_busy_thread_rejects_second_turn(...): ...
def test_failure_records_failed_and_error(...): ...
def test_awaiting_approval_then_decide(...): ...
def test_decide_when_not_awaiting_raises(...): ...
def test_recover_resumes_expired_running_thread_once(...):   # Review Focus 3
    # simulate crash: run until after "retrieve" via interrupt_after/partial stream, mark lease expired
    assert runner.recover() == [t] and runner.recover() == []   # second call finds nothing (lease claimed)
    assert runner.snapshot(t)["status"] == "done"
def test_recover_ignores_live_leases(...): ...
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 5: FastAPI service

**Files:**
- Create: `src/copilot/service/api.py`, `tests/test_api.py`
- Modify: `pyproject.toml` (`fastapi==0.146.0`, `uvicorn==0.54.0`, `sse-starlette==3.5.0`; dev: `httpx` already present)

**Interfaces:**
- `create_app(runner: ThreadRunner, settings: Settings) -> FastAPI`; lifespan calls `runner.recover()` on startup.
- Auth dependency: `X-API-Key` → role from `settings.api_keys`; missing/unknown → 401.
- Routes (all under `/api/v1`, all require a key):
  - `POST /threads` → `{"thread_id"}` (uuid4 hex[:8]); `GET /threads/{id}` → `runner.snapshot`.
  - `POST /threads/{id}/messages` `{"question": str (1–2000 chars)}` → 202; 409 if busy.
  - `GET /threads/{id}/events?after=N` → SSE stream of `events_since`, ending when status leaves `running`.
  - `POST /threads/{id}/approve` `{"approved": bool, "body": str | null}` → maintainer only (403 otherwise); 409 if not awaiting (covers double-click).
  - `GET /health` (no auth).
- `GET /` serves `static/index.html` (Task 6).

- [ ] **Step 1: Write the failing tests** (`TestClient`, runner with fake deps + `InlineExecutor`)

```python
def test_requires_api_key(client): ...                        # 401
def test_ask_and_poll_until_done(client): ...                 # 202 then snapshot done with answer + sources
def test_busy_returns_409(client): ...
def test_reader_cannot_approve(client): ...                   # 403, nothing executed (Review Focus 5)
def test_double_approve_second_is_409(client): ...            # Review Focus 1
def test_question_length_validated(client): ...               # 422
def test_events_stream_lists_nodes(client): ...
def test_startup_recovers_threads(...): ...
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 6: Chat page + `copilot serve` + CLI approvals

**Files:**
- Create: `src/copilot/service/static/index.html`
- Modify: `src/copilot/cli.py`, `tests/test_cli.py`, `tests/test_api.py`

**Interfaces:**
- `index.html` (no build step, no framework): API-key field (kept in `localStorage`), question box, live progress line from SSE (`planning → searching → checking GitHub → writing → verifying`), answer rendered as text with `[id]` linked to sources, a sources list, and — when `pending_action` exists — a card showing repo/target/body (editable textarea) with **Approve** / **Reject** buttons.
- `copilot serve [--host 127.0.0.1] [--port 8000]` → builds deps (gateway, reranker, read/write MCP clients opened once for the app's lifetime), `ThreadRunner`, `create_app`, runs uvicorn.
- `copilot ask` prints a pending action and accepts `--approve` / `--reject` with `--thread` to decide it from the terminal.

- [ ] **Step 1: Write the failing tests** — `GET /` returns HTML containing `id="question"`, `id="approve"`, and the SSE URL path; CLI `ask` with a fake requested action prints "Awaiting approval" and `ask --thread X --approve` executes once.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run --frozen pytest -q` → all PASS.

---

### Task 7: Real end-to-end run (needs author setup) and docs

- [ ] **Step 1: Author setup (ask the author first):** create a sandbox repo (e.g. `<you>/copilot-sandbox`) with one test issue; create a **fine-grained token for only that repo** with *Issues: Read and write*; set `COPILOT_SANDBOX_REPO`, `COPILOT_GITHUB_WRITE_TOKEN`, and `COPILOT_API_KEYS` in `.env`.
- [ ] **Step 2: Live lookup demo** (confirm spend first, < $0.01): "Is PR #3564 merged, and who opened it?" → answer cites the live PR (state closed, not merged, Kludex).
- [ ] **Step 3: HITL demo:** "Comment on issue #1 in the sandbox summarizing how lifespan works" → pending card in the chat page → edit → approve → comment visible on GitHub; click approve again → 409; check `actions` table has one row.
- [ ] **Step 4: Durability demo:** start a question, `kill -9` the server after the planner step (watch the events), restart `copilot serve` → thread resumes and finishes; record timings.
- [ ] **Step 5:** README: architecture diagram (tools + HITL + service), safety layers table, screenshots/GIF placeholders, demo commands; `docs/results/<date>-plan3-demo.md` with what happened.

## Done when

- `uv run --frozen pytest -q` green.
- Live lookup, approve-once HITL against the sandbox repo, and kill-9 auto-resume all demonstrated on real services.
- Total project spend still well under $1.
