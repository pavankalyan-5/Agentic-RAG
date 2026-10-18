# Dev Docs Copilot — Design

*Date: 2026-10-18 · Source spec: [spec.md](../../../spec.md)*

## 1. Intent and why

A portfolio project the author will **learn from, build, and put on their resume**. It turns the generic "enterprise agentic RAG platform" in `spec.md` into one concrete product:

> **Dev Docs Copilot** — an onboarding assistant that answers engineers' questions about a codebase using its docs, issues, PRs, and review discussions, cites every claim, and can take approved actions on GitHub.

**Why RAG is justified (the interview answer):** in real use it runs against a company's *private* repos, which no LLM was trained on, and the data changes daily. The portfolio demo uses a public repo only because private data can't be shown. The eval suite proves the point with a measured **with-RAG vs. no-RAG** comparison.

Example questions:
- "Why did they switch X to Y?" (answered from PR/issue discussion)
- "Is there an open bug about Z, and has anyone tried fixing it?"
- "How do I configure W?" (answered from docs)

## 2. Goals

1. A working end-to-end agent: question → plan → hybrid retrieval → live GitHub lookup → cited answer.
2. Cover all five phases of `spec.md` with real (not checkbox) use.
3. Produce **measured results** usable as resume bullets: retrieval hit rate per strategy, faithfulness, with/without-RAG delta, cost and latency per request.
4. Teach the author: retrieval, MCP, durable LangGraph agents, OpenTelemetry tracing, evals, guardrails. (Author already knows Python, FastAPI, Docker well and LangChain/LangGraph somewhat — don't spend effort explaining those basics.)
5. A polished repo: README with architecture diagram, results tables, demo GIF, one-command `docker compose up`.

## 3. Non-goals (cut from spec.md)

- **Jira** — GitHub alone carries the MCP story.
- **Qdrant** — Postgres only. (May appear later as an optional benchmark, not a dependency.)
- **Llama Guard** — needs GPU or another provider; replaced by lighter checks (see §6.6).
- **Redis** — Postgres covers short-term memory via checkpoints.

**Stretch goals (week 6, only if ahead):** long-term user memory (entity/preference extraction); online LLM-as-judge sampling of ~10% of traces.

## 4. Anti-goals (failure even if it technically works)

- A feature checklist with no numbers — every phase should yield something measurable.
- The agent posting to a real open-source project's issues during development or demos.
- Uncited or fabricated answers (e.g., invented issue numbers).
- Infrastructure so heavy that `docker compose up` on a laptop is painful.
- An eval suite that's too expensive to run on every PR.

## 5. Constraints

- **Timeline:** ~6 weeks, part-time (~10–15 h/week).
- **LLM provider:** OpenAI (author has a key).
- **Budget: $10 of OpenAI credit total for the whole project** (not per month). This is a hard constraint that shapes model choice and eval design:
  - Use the cheapest capable models (mini/nano tier) for **everything** by default, including the final answer and the eval judge; the stronger-model option stays in config but is off.
  - Embeddings are cheap; chat calls (especially eval judging) are the main cost.
  - **Cache LLM responses** (keyed on model + prompt) during development and eval runs, so re-runs of unchanged cases are free.
  - Every LLM call goes through one client wrapper that records cost (from Phase 3 traces) and enforces a **configurable spend ceiling** per run and in total; set a matching usage limit in the OpenAI dashboard.
  - The LLM client must be OpenAI-API-compatible and swappable via config (e.g., to a local Ollama model) so development can continue if credit runs low.
- **Runs locally** via Docker Compose on a laptop (no GPU assumed).
- **Not yet a git repo** — initialize one at the start; GitHub hosting is needed anyway for the Actions eval gate.

## 6. Architecture — decided parts

### 6.1 Storage: one PostgreSQL instance

Decided with the author: a single Postgres for everything.

| Store | Contents |
|---|---|
| `documents` | One row per source item (doc page, issue, PR, design doc) with metadata: repo, type, number, title, author, labels, state, URL, created/updated timestamps |
| `chunks` | Chunk text, FK to document, chunk ID, **pgvector** embedding, **BM25 index via ParadeDB `pg_search`** |
| LangGraph checkpoint tables | Created by `PostgresSaver` (checkpoints, writes) plus a `thread_metadata` table (owner, created_at, status) |
| `sync_state` | Per repo + source type: last successful sync timestamp |

*My call:* use the ParadeDB Postgres image, which bundles pgvector and pg_search.

### 6.2 Ingestion (offline index = history)

- Script/CLI pulls from the GitHub REST/GraphQL API: Markdown docs + README, issues + comments, PRs + description + review comments.
- **Type-aware chunking:** docs split on headings; an issue/PR thread is kept together (question and resolution in one chunk where size allows; long threads split with the title and number repeated in each chunk).
- Embeds with OpenAI's small embedding model (verify current model name/price at plan time).
- **Incremental sync** using `updated_at` / `since`; upsert changed documents and re-chunk them.
- *My call:* to bound cost and size, the demo index covers roughly the last 12 months of issues/PRs plus all current docs (pick a demo repo where that is a few thousand items at most). Configurable.

### 6.3 Live tools (MCP = current state)

- GitHub's official MCP server, run via stdio (in Docker), authenticated with a fine-grained PAT; connected through `langchain-mcp-adapters` (or equivalent).
- **Read tools** (issue status, recent merges, search) — callable freely.
- **Write tools** (e.g., comment on an issue, create an issue) — always gated by human approval (§6.5) and **pointed at a sandbox repo the author owns**, never the public demo repo.

### 6.4 Retrieval pipeline

`Decompose query (1–3 sub-queries) → per sub-query: dense (pgvector) + sparse (BM25) → Reciprocal Rank Fusion → cross-encoder rerank → top-K context`

- Metadata filters supported (type, state, date range) — the planner may set them.
- Reranker: a free, local open-source cross-encoder (CPU), e.g., a BGE reranker variant. *My call.*
- Must be switchable between strategies (dense-only, sparse-only, hybrid, hybrid+rerank) for the retrieval experiment (§7).

### 6.5 Agent runtime (LangGraph)

- **State:** `messages`, `plan`, `retrieved_docs`, `next_step`, `error_context` (per spec.md §1.3).
- **Nodes (indicative):** guard_input → planner → retrieve → tool_calls (MCP) → synthesize → verify_citations → guard_output. Planner decides whether retrieval and/or live tools are needed.
- **Durability:** `PostgresSaver` checkpoints every transition. *My call:* **automatic resume** — graph runs execute in a background runner (not tied to the HTTP request); `thread_metadata` tracks status (`running`, `interrupted`, `done`, `failed`) plus a lease/heartbeat; on startup the server resumes threads still marked `running` whose lease expired, so a thread is never run twice. The client reconnects to the thread to receive the result. This is the durability demo.
- **HITL:** any write tool triggers `interrupt`; `POST /api/v1/threads/{thread_id}/approve` (with approve/reject + optional edit) resumes.
- **Retries:** exponential backoff with jitter on OpenAI and GitHub calls for 429/5xx (LangGraph `RetryPolicy` and/or `tenacity`).
- **Models:** a cheap fast model for planning and judging, a stronger model for final synthesis; both set in config.

### 6.6 Guardrails

- **Input:** prompt-injection detection = rules/heuristics + a cheap LLM classifier; also applied to **retrieved content** (issues are user-written and can carry injections). *My call:* rules run on every chunk; the LLM classifier runs only on the user query and on chunks the rules flag, to bound cost.
- **PII redaction (Microsoft Presidio)** — *my call on scope:*
  - **Where:** at ingestion (before embedding, so the index never holds raw PII) **and** on the user's query at request time.
  - **What counts:** emails, phone numbers, credit cards, SSNs, IP addresses, API keys/tokens. **GitHub usernames and author names are not PII** here — they're public handles and core to "who worked on this?" answers.
  - **"Leak" =** any of those entity types present in an outbound OpenAI request payload, checked by a test that scans logged payloads over a seeded PII test set.
- **Output:** final answer is a Pydantic model (answer text + list of claims, each with chunk IDs/URLs); citation verifier rejects or regenerates answers citing IDs not in the retrieved set.
- **Authorization:** simple API-key → role mapping (`reader` can ask; `maintainer` can approve write actions). Enforced in the tool layer, not just the UI.
- Implemented as middleware: input scan before graph execution, output validation before the HTTP response.

### 6.7 Observability

- OpenTelemetry + OpenInference instrumentation on LLM calls, retrieval (dense and sparse as separate spans), MCP tool calls, and graph nodes; span tree as in spec.md §3.2.
- *My call:* export to **Arize Phoenix** (single container, OpenInference-native) rather than Langfuse, whose self-hosted v3 needs several extra services. Instrumentation stays vendor-neutral, so switching is a config change.
- Report per-node latency p50/p90/p99, tokens by model, and $/request.

### 6.8 Evals and CI gate

- **Golden dataset:** 50–100 cases — query, expected source documents/chunks, reference answer — plus adversarial injection cases. Generated with LLM help from the indexed corpus, then **hand-reviewed** by the author.
- *My call:* **DeepEval** (pytest-native, fits CI) for faithfulness, answer relevance, contextual precision/recall, hallucination; plus a plain retrieval hit-rate@K metric.
- *My call on live data:* the **gated PR set uses only questions answerable from the frozen index snapshot** (no MCP calls, reproducible). Questions that depend on live repo state run in the manual suite, judged on tool-use correctness and faithfulness to the tool output rather than a fixed reference answer.
- **GitHub Actions** on PRs: Postgres service container loaded from a **checked-in/artifact index snapshot** (no re-embedding in CI); comment scores on the PR; fail if faithfulness < 0.85.
  - *Budget-driven (my call):* **retrieval metrics (hit rate, MRR) run on every PR on the full golden set** — no LLM needed, free. **LLM-judged metrics run on a ~15-case subset**, only when PRs touch agent/retrieval/prompt code (path filters), using the cheapest judge model and the response cache.
  - Full suite runs **manually only** (no nightly schedule — it would drain the credit).
- OpenAI key provided as a repo secret.

### 6.9 API and UI

- FastAPI: chat endpoint, thread endpoints, approve endpoint, health.
- *My call on streaming:* stream **progress events** over SSE ("planning", "searching docs", "checking GitHub", "verifying citations"), then send the **final, validated answer** in one piece. Answer tokens aren't streamed, because output guardrails must pass before the user sees anything.
- A simple web chat page showing the answer, clickable citations, and approve/reject buttons for pending actions. Enough for a demo GIF — not a product UI.

## 7. Measured deliverables (resume numbers)

1. **Retrieval experiment:** hit rate@K and MRR for dense-only vs. BM25-only vs. hybrid vs. hybrid+rerank.
2. **With-RAG vs. no-RAG:** faithfulness/correctness on the golden set.
3. **Cost/latency:** $/request and p50/p90 before and after one optimization (e.g., cheaper planner model, caching).
4. **Guardrails:** injection block rate on the adversarial set; PII leak count (target zero).
5. **Durability demo:** kill the container mid-run, resume with no state loss.

## 8. Schedule (6 weeks)

| Week | Focus |
|---|---|
| 1 | Repo setup, Docker Compose, schema, GitHub ingestion + chunking + embeddings, first hybrid search + RRF |
| 2 | LangGraph agent: planner, reranker, Postgres checkpoints, retries, synthesize with citations, basic API |
| 3 | GitHub MCP (read + gated write), HITL approve endpoint, chat page — **end-to-end working** |
| 4 | OpenTelemetry/OpenInference → Phoenix; cost & latency reporting; golden dataset build |
| 5 | DeepEval suite, GitHub Actions PR gate, retrieval experiment, with/without-RAG comparison |
| 6 | Guardrails (injection, Presidio, citation verifier, roles); README, diagram, demo GIF, resume bullets; stretch goals if ahead |

## 9. Left to the builder

- Choice of the public demo repo: a mid-sized, active repo with substantial design discussion in issues/PRs (more interesting than one with polished docs only). Pick and confirm with the author in week 1.
- Exact chunk sizes, K values, RRF constant, prompts.
- Python packaging/tooling (e.g., uv), project layout, test structure.
- Front-end tech for the chat page (keep it minimal).
- Exact OpenAI model names (check current options and pricing).
