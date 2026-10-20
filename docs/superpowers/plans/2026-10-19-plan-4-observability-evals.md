# Plan 4 — Observability (Tracing) & Evaluations (Evals + CI Quality Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) Every agent turn produces a trace — a tree of timed steps (planner, retrieval, rerank, live lookups, LLM calls with tokens and cost) — viewable in Arize Phoenix, plus a latency/cost report. (2) A golden question set and an evaluation suite measure retrieval quality and answer quality, and (3) a GitHub Actions workflow runs the evals on every pull request, posts the scores as a PR comment, and fails the build if quality drops below thresholds. The results become the project's resume numbers.

**Architecture:** OpenTelemetry (vendor-neutral) with OpenInference conventions: auto-instrumentation for OpenAI and LangGraph, plus manual spans for retrieval, reranking, MCP tools, and gateway cost. Exported via OTLP to a Phoenix container. Evals live in `src/copilot/evals/`: a JSONL golden set keyed by **document ids** (stable across re-chunking), deterministic retrieval metrics (no LLM), and DeepEval answer metrics whose judge LLM is an adapter over **our gateway** (cache, ledger, budget ceiling all apply). CI restores a data snapshot (index + LLM cache) so unchanged prompts cost $0.

**Tech Stack:** opentelemetry-sdk 1.45.1, arize-phoenix-otel 0.17.2, openinference-instrumentation-openai 0.1.65, openinference-instrumentation-langchain 0.1.79, arize-phoenix-client 3.5.0 (+ pandas), Phoenix image `arizephoenix/phoenix:version-20.23.0`, deepeval 4.3.1, GitHub Actions.

**Spec:** [docs/superpowers/specs/2026-10-18-dev-docs-copilot-design.md](../specs/2026-10-18-dev-docs-copilot-design.md) §6.7 (observability), §6.8 (evals + CI), §7 (measured deliverables), §5 (budget).

**Verified 2026-10-19:** Phoenix 20.23.0 container receives OTLP/HTTP spans at `/v1/traces` and the Python client returns them with parent links (`phoenix.client.Client(...).spans.get_spans_dataframe`, needs pandas); DeepEval 4.3.1 custom judge = subclass `DeepEvalBaseLLM` implementing `generate`, `a_generate`, `get_model_name`, `load_model`; telemetry off via `DEEPEVAL_TELEMETRY_OPT_OUT=YES`.

## Global Constraints

- **Every LLM call — including eval judges — goes through `LLMGateway`** (cache + ledger + ceiling). DeepEval never gets its own OpenAI key path. `DEEPEVAL_TELEMETRY_OPT_OUT=YES` is set before DeepEval is imported.
- Tracing is **off unless `COPILOT_PHOENIX_ENDPOINT` is set**; tracing failures never break a turn. Prompts/answers in traces are already PII-redacted text (ingestion redaction); traces stay local (self-hosted Phoenix).
- Golden cases reference relevant sources by **document `source_id`** (e.g. `doc:docs/lifespan.md`, `pr:3564`), never chunk ids.
- Judge model: `gpt-4.1-mini` (configurable `eval_judge_model`). Eval agent runs use the normal agent with **no live tools and no write tools** (deterministic, offline-safe).
- Budget for plan 4 real runs: target < $1.50 total. CI runs set `COPILOT_BUDGET_USD` explicitly; identical prompts are cache hits.
- Quality gate (from spec): **faithfulness ≥ 0.85**; plus retrieval **hit_rate@5 ≥ 0.85** (spec Phase 2 target). Thresholds live in `evals/thresholds.json`.
- Tests: no Phoenix, no network, no OpenAI — spans captured with OpenTelemetry's in-memory exporter; judges/agents faked.

## Review Focus

1. **Tracing backend down or misconfigured** → turns still succeed; at most a log line. Test in Task 1.
2. **The eval suite spends money unexpectedly** (a judge call bypassing the gateway, or re-running unchanged cases) → impossible by construction: judge = gateway adapter; second identical run costs $0. Test in Task 3 / Task 6.
3. **Golden set rots** (re-index renumbers chunks; a doc file renamed) → cases keyed by `source_id`; the loader reports cases whose expected sources no longer exist instead of silently scoring 0. Test in Task 4.
4. **CI on a fork / without the OpenAI secret** → retrieval metrics still run and gate; LLM-judged metrics are skipped with a clear note, not a red build for the wrong reason. Test in Task 7.
5. **Flaky judge** (LLM judges vary) → temperature 0 + cache; gate uses averages over cases, not single cases; the report shows per-case reasons for every failure. Test in Task 6.

---

## File Structure

```
docker-compose.yml                 + phoenix service
src/copilot/
  config.py                        + phoenix_endpoint, eval_judge_model
  observability/__init__.py
  observability/tracing.py         setup_tracing(), span(), instrument helpers
  observability/report.py          compute_report(spans) -> per-node percentiles, tokens, $/turn
  llm/gateway.py                   + complete() (free-text, for judges); cost/cache span attributes
  retrieval/search.py, rerank.py   + RETRIEVER / RERANKER spans
  tools/mcp_client.py              + TOOL spans
  agent/graph.py, service/runner.py + root "agent_turn" span (session.id = thread)
  evals/__init__.py
  evals/dataset.py                 EvalCase, load_golden(), validate_against_index()
  evals/generate.py                draft_cases() — LLM-drafted questions per document (for human review)
  evals/judge.py                   GatewayJudge(DeepEvalBaseLLM)
  evals/retrieval_eval.py          hit_rate@k, MRR@k, recall@k per strategy
  evals/answer_eval.py             run agent on cases; DeepEval metrics; safety/unanswerable checks; no-RAG baseline
  evals/snapshot.py                export/import index + cache tables (COPY, version-independent)
  evals/report.py                  markdown/JSON reports + threshold gate
  cli.py                           + `copilot trace-report`, `copilot evals ...`
evals/golden.jsonl, evals/thresholds.json
.github/workflows/eval.yml
tests/test_tracing.py, test_trace_report.py, test_judge.py, test_eval_dataset.py,
      test_retrieval_eval.py, test_answer_eval.py, test_snapshot.py
```

---

### Task 1: Tracing

**Files:** create `observability/__init__.py`, `observability/tracing.py`, `tests/test_tracing.py`; modify `config.py`, `docker-compose.yml` (service `phoenix`: image `arizephoenix/phoenix:version-20.23.0`, ports `6006:6006`, volume for its data), `llm/gateway.py`, `retrieval/search.py`, `retrieval/rerank.py`, `tools/mcp_client.py`, `agent/graph.py` (`run_turn`/`resume_turn` root span), `service/runner.py` (root span around `_execute`), `pyproject.toml` (deps above).

**Interfaces:**
- Settings: `phoenix_endpoint: str = ""` (e.g. `http://127.0.0.1:6006/v1/traces`), `phoenix_project: str = "dev-docs-copilot"`.
- `setup_tracing(settings, *, exporter=None) -> TracerProvider | None` — no endpoint and no exporter → None (no-op). Otherwise builds a provider (BatchSpanProcessor for OTLP; SimpleSpanProcessor for a given test exporter), instruments OpenAI (`OpenAIInstrumentor`) and LangChain/LangGraph (`LangChainInstrumentor`), sets it global. Idempotent. Exporter errors are logged, never raised.
- `span(name, kind, **attrs)` context manager using OpenInference attribute names (`openinference.span.kind` ∈ CHAIN/RETRIEVER/RERANKER/TOOL/LLM; `input.value`, `output.value`, `session.id`, `retrieval.documents.{i}.document.id/score/content` — content truncated to 300 chars, max 10 docs).
- Instrumented points: `search()` (RETRIEVER, attrs `copilot.mode`, `copilot.k`), `rerank()` (RERANKER, scores), `MCPToolClient.call` (TOOL, `tool.name`, outcome), gateway `chat/embed/complete` (sets `copilot.cost_usd`, `copilot.cache_hit`, `llm.model_name` on the current span), root `agent_turn` (CHAIN, `session.id` = thread id, input = question, output = answer text, `copilot.status`).

- [ ] **Step 1: Write the failing tests** (InMemorySpanExporter; fake deps from plan 2)

```python
def test_turn_produces_a_span_tree(traced_graph, exporter):
    run_turn(traced_graph, "t1", "What is ERR_TIMEOUT?")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["agent_turn"]
    assert root.attributes["session.id"] == "t1" and root.attributes["openinference.span.kind"] == "CHAIN"
    for name in ("plan", "retrieve", "synthesize", "verify", "search", "rerank"):
        assert name in spans                               # node spans come from the LangChain instrumentor
    assert all(s.context.trace_id == root.context.trace_id for s in spans.values())

def test_retriever_span_lists_documents(...): ...          # retrieval.documents.0.document.id present, content truncated
def test_gateway_marks_cost_and_cache(...): ...            # second identical chat -> copilot.cache_hit True, cost 0
def test_tool_span_records_failure(...): ...               # ToolError -> span status ERROR, turn still answers
def test_tracing_disabled_without_endpoint(settings): assert setup_tracing(settings) is None
def test_exporter_failure_never_breaks_a_turn(...): ...    # exporter raising on export -> run_turn still returns
```

- [ ] **Step 2:** run → FAIL. **Step 3:** `UV_HTTP_TIMEOUT=300 uv add` the deps; implement. **Step 4:** `uv run --frozen pytest -q` → PASS.
- [ ] **Step 5 (manual):** `docker compose up -d phoenix`; set `COPILOT_PHOENIX_ENDPOINT`; ask one cached question via `copilot ask`; open http://127.0.0.1:6006 and confirm the tree. ($0 — cached.)

---

### Task 2: Latency & cost report from traces

**Files:** create `observability/report.py`, `tests/test_trace_report.py`; modify `cli.py`.

**Interfaces:**
- `compute_report(spans: list[dict]) -> dict` — input rows with `name`, `span_kind`, `start_time`, `end_time`, `trace_id`, `attributes` (Phoenix dataframe rows or test dicts). Output: `{"turns": n, "latency_s": {node: {"p50", "p90", "p99", "count"}}, "turn_latency_s": {...}, "tokens_by_model": {model: {"prompt", "completion"}}, "cost_usd": {"total", "per_turn_mean", "per_turn_p90"}, "cache_hit_rate": float}`. Percentiles: nearest-rank method (documented).
- `copilot trace-report [--since-hours 24] [--out docs/results/<date>-trace-report.md]` — fetches spans from Phoenix (`phoenix.client`), prints a table, optionally writes markdown.

- [ ] **Step 1: Write the failing tests** — hand-built span rows: percentiles on `[1..10]` → p50=5, p90=9, p99=10; tokens summed per model; per-turn cost = sum of `copilot.cost_usd` within a trace; turns with zero LLM spans handled; empty input → zeros, no crash.
- [ ] **Step 2–4:** FAIL → implement → PASS.

---

### Task 3: Judge adapter over the gateway

**Files:** create `evals/__init__.py`, `evals/judge.py`, `tests/test_judge.py`; modify `llm/gateway.py`, `config.py`, `pyproject.toml` (`deepeval==4.3.1`).

**Interfaces:**
- `LLMGateway.complete(messages, *, model, operation) -> str` — free-text completion with the same cache key scheme (no schema), ledger and ceiling.
- `class GatewayJudge(DeepEvalBaseLLM)` — `__init__(gateway, model)`; `generate(prompt, schema=None)` → `gateway.chat(..., schema=schema, operation="judge")` when a schema is given, else `gateway.complete(..., operation="judge")`; `a_generate` delegates to `generate` (metrics run with `async_mode=False`); `get_model_name()` → model; `load_model()` → self.
- `evals/__init__.py` sets `os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")` before importing deepeval anywhere.
- Settings: `eval_judge_model: str = "gpt-4.1-mini"`.

- [ ] **Step 1: Write the failing tests** — `complete` caches identical prompts and records `operation="judge"` with cost; `GatewayJudge.generate` with and without a schema; a second identical call is free; budget ceiling blocks judge calls; telemetry env var set after importing `copilot.evals`.
- [ ] **Step 2–4:** FAIL → implement → PASS.

---

### Task 4: Golden dataset (format, validation, drafting)

**Files:** create `evals/dataset.py`, `evals/generate.py`, `tests/test_eval_dataset.py`, `evals/thresholds.json`; modify `cli.py`.

**Interfaces:**
- `class EvalCase(BaseModel)`: `id: str`, `category: Literal["docs", "history", "multi_part", "unanswerable", "adversarial"]`, `question: str`, `expected_sources: list[str]` (document `source_id`s; empty for unanswerable/adversarial), `reference_answer: str | None`, `must_not: list[str] = []` (substrings the answer must not contain, e.g. for injection cases), `in_ci: bool = False` (the ~15-case PR subset), `notes: str = ""`.
- `load_golden(path) -> list[EvalCase]` (JSONL; duplicate ids rejected). `validate_against_index(conn, cases, repo) -> list[str]` — problems such as "case docs-07: expected source doc:docs/old.md not in index".
- `draft_cases(conn, chat, *, repo, per_category: dict[str, int], seed: int) -> list[EvalCase]` — samples documents (docs pages; PRs with discussion) deterministically; asks `answer_model` (operation `"eval_draft"`) for one natural question + a reference answer **answerable only from that document**; also drafts multi-part questions from document pairs. Unanswerable and adversarial cases come from a hand-written seed list in `evals/seed_cases.jsonl`. Output goes to `evals/golden_draft.jsonl` **for human review**; `golden.jsonl` is written only by the author.
- `copilot evals draft [--n 60]` and `copilot evals check` (load + validate).
- `evals/thresholds.json`: `{"faithfulness": 0.85, "hit_rate@5": 0.85, "answer_relevancy": 0.80, "unanswerable_correct": 0.80, "adversarial_safe": 1.0}`.

- [ ] **Step 1: Write the failing tests** — round-trip JSONL; duplicate id rejected; unknown category rejected; validation flags a missing source; `draft_cases` with `FakeChat` produces cases whose `expected_sources` are the sampled documents, deterministic for a fixed seed.
- [ ] **Step 2–4:** FAIL → implement → PASS.

---

### Task 5: Retrieval evaluation (no LLM)

**Files:** create `evals/retrieval_eval.py`, `tests/test_retrieval_eval.py`; modify `cli.py`.

**Interfaces:**
- `retrieval_metrics(ranked_source_ids: list[list[str]], expected: list[set[str]], k: int) -> dict` → `{"hit_rate@k", "mrr@k", "recall@k"}`; ranked lists are **document-level** (chunks mapped to their document's `source_id`, de-duplicated preserving rank).
- `evaluate_retrieval(conn, embedder, reranker, cases, *, k=5) -> dict[strategy, metrics]` for strategies `dense`, `sparse`, `hybrid`, `hybrid+rerank` (question used directly as one query — isolates retrieval from the planner). Skips cases without expected sources.
- `copilot evals retrieval [--k 5] [--out ...]`.

- [ ] **Step 1: Write the failing tests** — metric math on hand-made rankings (hit/MRR/recall exact values, ties, k cut-off, de-dup); `evaluate_retrieval` on `seeded_db` returns all four strategies with values in [0, 1].
- [ ] **Step 2–4:** FAIL → implement → PASS.

---

### Task 6: Answer evaluation, safety checks, no-RAG baseline, gate

**Files:** create `evals/answer_eval.py`, `evals/report.py`, `tests/test_answer_eval.py`; modify `cli.py`.

**Interfaces:**
- `run_cases(graph_factory, cases) -> list[CaseRun]` — each case in a fresh thread; records answer text, sources (`source_id`s via document lookup), retrieved contexts (chunk texts), insufficient flag, pending action, latency, cost delta.
- `score_runs(runs, cases, judge, *, metrics=("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall", "correctness")) -> list[CaseScore]` — DeepEval metrics with `async_mode=False`, `model=judge`, `include_reason=True`; `correctness` = `GEval` vs `reference_answer`. Category rules: unanswerable → pass iff `insufficient_context` (no judge needed); adversarial → pass iff no `must_not` substring in the answer and no pending action targeting anything but the sandbox and no action at all unless the case expects one.
- `no_rag_baseline(chat, cases, model) -> list[str]` — the answer model with no context, same question; scored with `correctness` only.
- `summarize(scores) -> dict` (means, pass rates, hallucination rate = 1 − mean faithfulness) and `gate(summary, thresholds) -> list[str]` (failures; empty = pass).
- `render_markdown(summary, scores, ...) -> str` — summary table, per-category table, and **every failing case with the judge's reason**.
- `copilot evals run [--ci-only] [--no-judge] [--baseline] [--out ...]` — exit code 1 if `gate` fails.

- [ ] **Step 1: Write the failing tests** — with a `FakeMetric` seam (`score_runs(..., metric_factory=...)`) so DeepEval itself isn't called in unit tests: summary math; gate pass/fail at boundaries; unanswerable and adversarial rules; report lists failing cases with reasons; second run of the same cases makes zero new LLM calls (fake gateway counts).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5 (slow, real, ~$0.01):** `uv run pytest -m slow -k deepeval` — one real DeepEval Faithfulness call through `GatewayJudge` on a tiny case, proving the adapter works with the real library.

---

### Task 7: Data snapshot + GitHub Actions quality gate

**Files:** create `evals/snapshot.py`, `tests/test_snapshot.py`, `.github/workflows/eval.yml`; modify `cli.py`.

**Interfaces:**
- `export_snapshot(conn, path)` / `import_snapshot(conn, path)` — gzip of `COPY … TO STDOUT` for `documents, chunks, sync_state, llm_cache, llm_usage` (version-independent, no pg_dump); import truncates then `COPY FROM STDIN` in dependency order and resets sequences.
- `copilot evals snapshot export|import PATH`; the snapshot is published as a GitHub **release asset** (`eval-snapshot`) via `gh release upload`.
- Workflow `eval.yml`: on `pull_request` (paths: `src/**`, `evals/**`, `pyproject.toml`, `uv.lock`) + `workflow_dispatch`; Postgres service `paradedb/paradedb:0.26.2-pg18`; `astral-sh/setup-uv` pinned; cache uv + fastembed model dir; download snapshot (`gh release download eval-snapshot`); `copilot init-db`; import snapshot; `copilot evals retrieval`; if `secrets.COPILOT_OPENAI_API_KEY` present → `copilot evals run --ci-only` with `COPILOT_BUDGET_USD` = snapshot spend + 0.50; else → note "LLM-judged metrics skipped (no secret)"; post/update one PR comment with the markdown report (`gh pr comment --edit-last` or create); fail on gate.

- [ ] **Step 1: Write the failing tests** — snapshot round-trip into a second test database preserves row counts, embeddings, and sequences (new inserts don't collide); workflow file is valid YAML with the required steps/keys (parsed in a test); the "no secret" path in the CLI exits 0 with the skip note when only retrieval gates run.
- [ ] **Step 2–4:** FAIL → implement → PASS.

---

### Task 8: Real runs — golden set, experiments, gate on a real PR

- [ ] **Step 1 (author):** review `evals/golden_draft.jsonl` (≈60 drafted + ≈12 seed cases): fix or delete bad questions, check expected sources, mark ~15 `in_ci`; save as `evals/golden.jsonl`. (Cost to draft: ~$0.03.)
- [ ] **Step 2:** `copilot evals retrieval` → strategy table (dense / sparse / hybrid / hybrid+rerank) — **resume number #1**. ($0)
- [ ] **Step 3:** `copilot evals run --baseline` on the full set → faithfulness, relevancy, contextual precision/recall, correctness RAG vs no-RAG, unanswerable/adversarial pass rates — **resume numbers #2–#4**. Confirm spend first (~$0.5–1.0). Re-run once to show $0 on the second run.
- [ ] **Step 4:** `copilot trace-report` over the eval traces → p50/p90/p99 per node, tokens by model, $/turn.
- [ ] **Step 5:** export + upload the snapshot; author adds the `COPILOT_OPENAI_API_KEY` repo secret; open a small PR (e.g. a prompt tweak) → CI comment appears with scores; demonstrate a deliberately bad change (e.g. `rerank_keep=1`) failing the gate.
- [ ] **Step 6:** `docs/results/<date>-evals.md` + README "Results" section with the numbers and how they were measured.

## Done when

- `uv run --frozen pytest -q` green; traces visible in Phoenix with the full span tree.
- A reviewed golden set, a retrieval strategy comparison, RAG-vs-no-RAG numbers, and a latency/cost report exist in `docs/results/`.
- A real PR shows the eval comment, and a deliberately bad change fails the gate.
- Total project spend still well under $3.
