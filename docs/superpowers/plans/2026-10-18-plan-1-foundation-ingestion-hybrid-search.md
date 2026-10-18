# Plan 1 — Foundation, Ingestion & Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Postgres index of one GitHub repo's docs, issues, and PRs (PII-redacted, chunked, embedded) plus a `copilot search` CLI that returns dense, sparse (BM25), or RRF-fused hybrid results.

**Architecture:** One ParadeDB Postgres container holds documents, chunks (pgvector embeddings + pg_search BM25 index), sync state, and an LLM cost ledger/cache. A Python package (`copilot`) ingests from the GitHub REST API and a shallow git clone, redacts PII with Presidio, chunks by source type, embeds through a budget-enforcing gateway, and serves hybrid search.

**Tech Stack:** Python 3.12, uv, psycopg 3 + pgvector, ParadeDB `paradedb/paradedb:0.26.2-pg18` (pgvector 0.8.6, pg_search 0.26.2), OpenAI `text-embedding-3-small`, httpx + tenacity, Presidio + spaCy `en_core_web_sm`, tiktoken, pydantic-settings, Typer, pytest + respx.

**Spec:** [docs/superpowers/specs/2026-10-18-dev-docs-copilot-design.md](../specs/2026-10-18-dev-docs-copilot-design.md) (§5 constraints, §6.1 storage, §6.2 ingestion, §6.4 retrieval up to RRF, §6.6 PII scope).

**This is plan 1 of 5.** Later plans: (2) LangGraph agent + reranker + query decomposition + durable runtime; (3) GitHub MCP + HITL + API + chat page; (4) observability + evals + CI gate + experiments; (5) guardrails + polish.

## Global Constraints

- Python **3.12**; dependencies managed with **uv** (`pyproject.toml` + `uv.lock`).
- Database image pinned to **`paradedb/paradedb:0.26.2-pg18`**; host port **5433** (avoid clashing with a local Postgres).
- Embeddings: **`text-embedding-3-small`, 1536 dimensions**, priced at **$0.02 per 1M input tokens** (verify on platform.openai.com/pricing before first real run).
- **Total OpenAI budget is $10**; default spend ceiling `COPILOT_BUDGET_USD=8.00`. Every OpenAI call goes through `LLMGateway`; no other module imports `openai`.
- BM25 queries **must use `paradedb.match(...)`**, never the raw `content @@@ '<user string>'` form (the raw parser errors on `?`, `(`, `"`, `AND:` — verified). `key_field` is deprecated in 0.26 — omit it.
- PII entities redacted: `EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS, SECRET`. **Usernames and person names are NOT redacted.** Replacement token format: `<ENTITY_TYPE>` (e.g. `<EMAIL_ADDRESS>`).
- Redaction happens **before** chunking/embedding; raw PII never lands in `documents` or `chunks`.
- Demo index window default: **last 12 months** of issues/PRs + all current Markdown docs.
- Tests run against a real Postgres (`docker compose up -d db`); no live network in tests (GitHub via respx, OpenAI via fakes).

## Review Focus

1. **Sync fails midway** (GitHub 403/429, network drop, or `BudgetExceeded`) → documents already processed stay committed, `sync_state` is **not** advanced, and the next run picks up again. Test in Task 8.
2. **Re-running sync on unchanged data** → no duplicate chunks and **zero embedding calls**. Test in Task 8.
3. **Null/empty bodies and giant unbroken text** (issue with `"body": null`; a 20k-token paragraph with no newlines) → no crash; no chunk exceeds the token limit. Tests in Tasks 5 and 6.
4. **Natural-language queries with punctuation** (`why? (JWT) "auth" AND: #4521`) → results or empty list, never a SQL error. Test in Task 9.
5. **Look-alike non-PII** (`#4521`, `v0.110.0`, `@tiangolo`, a person's name) → left untouched by redaction. Test in Task 4.

---

## File Structure

```
pyproject.toml, uv.lock, .env.example, .gitignore, docker-compose.yml, README.md
src/copilot/
  config.py              Settings (pydantic-settings)
  db/schema.sql          All DDL (idempotent)
  db/connection.py       connect(), apply_schema()
  llm/pricing.py         PRICES table + cost_usd()
  llm/gateway.py         LLMGateway: cache, usage ledger, budget ceiling, embed()
  ingest/models.py       SourceDoc, Post dataclasses
  ingest/redact.py       redact()
  ingest/chunking.py     Chunk, count_tokens(), chunk_markdown(), chunk_thread()
  ingest/github_api.py   GitHubClient (issues, PRs, comments)
  ingest/docs_repo.py    iter_docs() via shallow git clone
  ingest/pipeline.py     ingest_documents(), sync_repo()
  retrieval/models.py    Hit, SearchFilters
  retrieval/search.py    dense_search(), sparse_search(), rrf_fuse(), search()
  cli.py                 Typer app: init-db, sync, search
tests/
  conftest.py            test DB fixtures
  fakes.py               FakeEmbedder, FakeOpenAI
  test_*.py              one per module
```

---

### Task 1: Project scaffold, Docker Compose, settings

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `docker-compose.yml`, `src/copilot/__init__.py`, `src/copilot/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `copilot.config.Settings` with fields `database_url: str`, `openai_api_key: str = ""`, `github_token: str = ""`, `embedding_model: str = "text-embedding-3-small"`, `embedding_dim: int = 1536`, `budget_usd: float = 8.0`, `demo_repo: str = ""`, `since_months: int = 12`; env prefix `COPILOT_`; reads `.env`. `get_settings() -> Settings` (cached).

- [ ] **Step 1:** `git init`, create `.gitignore` (Python, `.env`, `.venv`, `data/`), and `uv init --package --name copilot --python 3.12`. Add deps: `psycopg[binary] pgvector openai httpx tenacity presidio-analyzer presidio-anonymizer spacy tiktoken pydantic-settings typer`; dev deps: `pytest respx`. Add the spaCy model as a direct-URL dependency matching the installed spaCy minor version (e.g. `en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl`).
- [ ] **Step 2:** `docker-compose.yml` with service `db`: image `paradedb/paradedb:0.26.2-pg18`, env `POSTGRES_USER=copilot POSTGRES_PASSWORD=copilot POSTGRES_DB=copilot`, port `5433:5432`, named volume, healthcheck `pg_isready -U copilot`. `.env.example` lists every `COPILOT_*` var with `COPILOT_DATABASE_URL=postgresql://copilot:copilot@localhost:5433/copilot`.
- [ ] **Step 3: Write the failing test**

```python
def test_settings_reads_prefixed_env(monkeypatch):
    monkeypatch.setenv("COPILOT_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("COPILOT_BUDGET_USD", "3.5")
    s = Settings(_env_file=None)
    assert s.database_url == "postgresql://x/y"
    assert s.budget_usd == 3.5
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
```

- [ ] **Step 4:** Run `uv run pytest tests/test_config.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement `Settings` / `get_settings()` in `src/copilot/config.py`.
- [ ] **Step 6:** Run `uv run pytest tests/test_config.py -v` → PASS. Run `docker compose up -d db` and `docker compose ps` → `db` is `healthy`.
- [ ] **Step 7: Commit** — `git add -A && git commit -m "chore: scaffold project, compose db, settings"`

---

### Task 2: Schema and test database fixtures

**Files:**
- Create: `src/copilot/db/__init__.py`, `src/copilot/db/schema.sql`, `src/copilot/db/connection.py`, `tests/conftest.py`, `tests/test_schema.py`

**Interfaces:**
- Produces: `connect(url: str) -> psycopg.Connection` (registers pgvector types, `row_factory=dict_row`); `apply_schema(conn) -> None` (idempotent). Fixture `db` (a `psycopg.Connection` to `copilot_test`, all tables truncated before each test). Fixture `settings` (a fresh `Settings(_env_file=None, database_url=<test url>)` per test, mutable).
- Tables (exact columns later tasks rely on):
  - `documents(id bigserial PK, repo text NOT NULL, source_id text NOT NULL, doc_type text NOT NULL CHECK (doc_type IN ('doc','issue','pr')), number int, title text NOT NULL, url text NOT NULL, author text, labels text[] NOT NULL DEFAULT '{}', state text, created_at timestamptz, updated_at timestamptz NOT NULL, content_hash text NOT NULL, UNIQUE(repo, source_id))`
  - `chunks(id bigserial PK, document_id bigint NOT NULL REFERENCES documents ON DELETE CASCADE, chunk_index int NOT NULL, heading text, content text NOT NULL, doc_type text NOT NULL, embedding vector(1536) NOT NULL, UNIQUE(document_id, chunk_index))` + HNSW index `vector_cosine_ops` + `CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks USING bm25 (id, content, doc_type)`
  - `sync_state(repo text, source text, last_synced_at timestamptz NOT NULL, PRIMARY KEY(repo, source))`
  - `llm_cache(key text PK, model text NOT NULL, response jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now())`
  - `llm_usage(id bigserial PK, at timestamptz NOT NULL DEFAULT now(), model text NOT NULL, operation text NOT NULL, input_tokens int NOT NULL, output_tokens int NOT NULL DEFAULT 0, cost_usd numeric(12,6) NOT NULL, cached bool NOT NULL)`

- [ ] **Step 1: Write the failing tests** — `conftest.py`: session fixture connects to `COPILOT_TEST_ADMIN_URL` (default `postgresql://copilot:copilot@localhost:5433/copilot`), drops/creates `copilot_test`, applies schema; function fixture `db` truncates `documents, chunks, sync_state, llm_cache, llm_usage RESTART IDENTITY CASCADE`.

```python
def test_schema_creates_tables(db):
    rows = db.execute("select tablename from pg_tables where schemaname='public'").fetchall()
    assert {"documents","chunks","sync_state","llm_cache","llm_usage"} <= {r["tablename"] for r in rows}

def test_apply_schema_is_idempotent(db):
    apply_schema(db); apply_schema(db)

def test_bm25_and_vector_queries_work(db):
    # insert one document + chunk with embedding [1.0]+[0.0]*1535, content 'ERR_TIMEOUT raised'
    # assert a paradedb.match('content','ERR_TIMEOUT') query returns its id
    # assert ORDER BY embedding <=> query_vec LIMIT 1 returns its id
```

- [ ] **Step 2:** `uv run pytest tests/test_schema.py -v` → FAIL.
- [ ] **Step 3:** Write `schema.sql` (starts with `CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_search;`, all `IF NOT EXISTS`); implement `connect` and `apply_schema` (read the SQL via `importlib.resources`, execute, commit).
- [ ] **Step 4:** `uv run pytest tests/test_schema.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(db): schema with pgvector + bm25, test fixtures"` (add new files first).

---

### Task 3: LLM gateway — pricing, cache, usage ledger, budget ceiling, embeddings

**Files:**
- Create: `src/copilot/llm/__init__.py`, `src/copilot/llm/pricing.py`, `src/copilot/llm/gateway.py`, `tests/fakes.py`, `tests/test_gateway.py`

**Interfaces:**
- Produces:
  - `PRICES: dict[str, dict[str, float]]` (USD per 1M tokens), containing `"text-embedding-3-small": {"input": 0.02, "output": 0.0}`; `cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float` (raises `KeyError` for unknown model — no silent $0).
  - `class BudgetExceeded(Exception)`.
  - `class Embedder(Protocol): def embed(self, texts: list[str]) -> list[list[float]]`.
  - `class LLMGateway` — `__init__(self, conn, settings: Settings, client=None)` (`client` defaults to `openai.OpenAI(api_key=settings.openai_api_key)`); `spent_usd() -> float` (sum of `llm_usage.cost_usd`); `embed(texts: list[str]) -> list[list[float]]` (satisfies `Embedder`).
  - `tests/fakes.py`: `FakeEmbedder` (deterministic hashing-trick bag-of-words, 1536-dim, L2-normalized, counts calls in `.calls`), `FakeOpenAI` (mimics `client.embeddings.create(model=, input=)` returning objects with `.data[i].embedding` and `.usage.prompt_tokens`; counts calls).
- Plan 2 will add `LLMGateway.chat(...)` using the same cache/ledger/ceiling.

- [ ] **Step 1: Write the failing tests**

```python
def test_cost_usd_embedding():
    assert cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)

def test_embed_records_usage_and_caches(db, settings):
    fake = FakeOpenAI(); gw = LLMGateway(db, settings, client=fake)
    v1 = gw.embed(["hello", "world"]); v2 = gw.embed(["hello"])
    assert fake.calls == 1                      # second call fully cached
    assert v2[0] == v1[0] and len(v1[0]) == 1536
    rows = db.execute("select cached from llm_usage order by id").fetchall()
    assert [r["cached"] for r in rows] == [False, True]

def test_embed_only_sends_uncached_texts(db, settings):
    fake = FakeOpenAI(); gw = LLMGateway(db, settings, client=fake)
    gw.embed(["a"]); gw.embed(["a", "b"])
    assert fake.last_input == ["b"]

def test_budget_ceiling_blocks_calls(db, settings):
    settings.budget_usd = 0.001
    db.execute("insert into llm_usage(model,operation,input_tokens,cost_usd,cached) values ('x','embed',0,0.001,false)")
    with pytest.raises(BudgetExceeded):
        LLMGateway(db, settings, client=FakeOpenAI()).embed(["new text"])
```

- [ ] **Step 2:** `uv run pytest tests/test_gateway.py -v` → FAIL.
- [ ] **Step 3:** Implement. Cache key = `sha256(f"{model}\0{text}")`; batch uncached texts in groups of **96**; check `spent_usd() >= settings.budget_usd` before each API batch → `BudgetExceeded`; record one `llm_usage` row per call (cached hits recorded with `cost_usd=0, cached=true`); wrap the API call with tenacity (retry on `openai.RateLimitError`, `openai.APIConnectionError`, `openai.InternalServerError`; exponential backoff with jitter, max 5 attempts).
- [ ] **Step 4:** `uv run pytest tests/test_gateway.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(llm): budget-enforcing gateway with cache and usage ledger"`

---

### Task 4: PII redaction

**Files:**
- Create: `src/copilot/ingest/__init__.py`, `src/copilot/ingest/redact.py`, `tests/test_redact.py`

**Interfaces:**
- Produces: `redact(text: str) -> str` — replaces each detected entity with `<ENTITY_TYPE>`; `REDACTED_ENTITIES: tuple[str, ...]` = the six types in Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("raw,token", [
    ("mail me at jane.doe@example.com", "<EMAIL_ADDRESS>"),
    ("SSN 078-05-1120 here", "<US_SSN>"),
    ("card 4111 1111 1111 1111", "<CREDIT_CARD>"),
    ("call +1 415 555 2671", "<PHONE_NUMBER>"),
    ("server at 10.20.30.40", "<IP_ADDRESS>"),
    ("token ghp_" + "a"*36, "<SECRET>"),
    ("key sk-proj-" + "B"*40, "<SECRET>"),
    ("aws AKIA" + "Q"*16, "<SECRET>"),
])
def test_redacts(raw, token):
    out = redact(raw)
    assert token in out

@pytest.mark.parametrize("safe", [
    "see #4521 and v0.110.0", "thanks @tiangolo", "Sebastián Ramírez fixed it", "",
])
def test_leaves_non_pii_untouched(safe):
    assert redact(safe) == safe
```

- [ ] **Step 2:** `uv run pytest tests/test_redact.py -v` → FAIL.
- [ ] **Step 3:** Implement with a module-level (lazily built, cached) `AnalyzerEngine` using `NlpEngineProvider` configured for `en_core_web_sm`, plus a custom `PatternRecognizer` named `SECRET` with patterns for `ghp_|gho_|ghs_|github_pat_` tokens, `sk-[A-Za-z0-9_-]{20,}`, and `AKIA[0-9A-Z]{16}`. Call `analyze(..., entities=list(REDACTED_ENTITIES), language="en")` and anonymize with `OperatorConfig("replace", {"new_value": f"<{type}>"})` per entity. Return early for empty strings.
- [ ] **Step 4:** `uv run pytest tests/test_redact.py -v` → PASS. If a phone/IP case is flaky, adjust `score_threshold` (not the test).
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): Presidio PII redaction incl. secrets"`

---

### Task 5: Chunking

**Files:**
- Create: `src/copilot/ingest/models.py`, `src/copilot/ingest/chunking.py`, `tests/test_chunking.py`

**Interfaces:**
- Produces (`models.py`):
  - `@dataclass Post: author: str | None; body: str; created_at: datetime | None`
  - `@dataclass SourceDoc: repo: str; source_id: str; doc_type: Literal["doc","issue","pr"]; number: int | None; title: str; url: str; author: str | None; labels: list[str]; state: str | None; created_at: datetime | None; updated_at: datetime; body: str; posts: list[Post]` — `source_id` format: `"doc:<path>"`, `"issue:<n>"`, `"pr:<n>"`.
- Produces (`chunking.py`):
  - `@dataclass Chunk: index: int; text: str; heading: str | None`
  - `count_tokens(text: str) -> int` (tiktoken `cl100k_base`)
  - `chunk_markdown(text: str, max_tokens: int = 500) -> list[Chunk]` — split at `#`–`###` headings (heading carried in `Chunk.heading` and prefixed to text), then pack paragraphs; a paragraph over the limit is hard-split by tokens.
  - `chunk_thread(doc: SourceDoc, max_tokens: int = 800) -> list[Chunk]` — renders `"#<n> <title>\n\n<body>\n\n<author>: <post>..."`; one chunk if it fits; otherwise packs whole posts, and **every** chunk starts with the `"#<n> <title>"` header line; oversized single posts are hard-split.

- [ ] **Step 1: Write the failing tests**

```python
def test_markdown_splits_on_headings():
    chunks = chunk_markdown("# A\nalpha\n## B\nbeta")
    assert [c.heading for c in chunks] == ["A", "B"]

def test_no_chunk_exceeds_limit_even_without_newlines():
    big = "word " * 20_000
    assert all(count_tokens(c.text) <= 500 for c in chunk_markdown(big, max_tokens=500))

def test_thread_kept_whole_when_small(issue_doc):
    assert len(chunk_thread(issue_doc)) == 1

def test_long_thread_repeats_header(long_issue_doc):
    chunks = chunk_thread(long_issue_doc, max_tokens=200)
    assert len(chunks) > 1 and all(c.text.startswith("#123 Background tasks broken") for c in chunks)
    assert all(count_tokens(c.text) <= 200 for c in chunks)

def test_empty_body_thread_still_chunks_title(empty_issue_doc):
    assert chunk_thread(empty_issue_doc)[0].text.startswith("#7 ")

def test_empty_markdown_gives_no_chunks():
    assert chunk_markdown("   ") == []
```

(Fixtures build `SourceDoc`s inline; `long_issue_doc` has 30 posts of ~60 words each.)

- [ ] **Step 2:** `uv run pytest tests/test_chunking.py -v` → FAIL.
- [ ] **Step 3:** Implement models and chunking as specified.
- [ ] **Step 4:** `uv run pytest tests/test_chunking.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): type-aware chunking"`

---

### Task 6: GitHub API source (issues, PRs, comments)

**Files:**
- Create: `src/copilot/ingest/github_api.py`, `tests/test_github_api.py`

**Interfaces:**
- Consumes: `SourceDoc`, `Post` (Task 5).
- Produces: `class GitHubClient` — `__init__(self, token: str, http: httpx.Client | None = None, base_url: str = "https://api.github.com")`; `iter_threads(repo: str, since: datetime) -> Iterator[SourceDoc]` — yields issues **and** PRs updated at/after `since`, ascending by `updated_at`, each with its comments as `posts` (PRs also include review comments from `/pulls/{n}/comments`, merged and sorted by `created_at`).

- [ ] **Step 1: Write the failing tests** (respx mocks)

```python
def test_yields_issue_and_pr_with_comments(respx_mock):
    # /repos/o/r/issues?state=all&since=...&sort=updated&direction=asc&per_page=100
    #   -> [issue #1 (body None), item #2 with "pull_request" key]
    # /issues/1/comments -> 1 comment; /issues/2/comments -> []; /pulls/2/comments -> 1 review comment
    docs = list(GitHubClient("t", http=httpx.Client()).iter_threads("o/r", SINCE))
    assert [(d.doc_type, d.number) for d in docs] == [("issue", 1), ("pr", 2)]
    assert docs[0].body == "" and len(docs[0].posts) == 1
    assert docs[1].source_id == "pr:2" and len(docs[1].posts) == 1

def test_follows_link_pagination(respx_mock): ...   # page 1 has Link rel="next"; assert both pages' items yielded

def test_retries_on_429_then_succeeds(respx_mock):
    # first response 429 with Retry-After: 0, second 200 -> assert one doc yielded, route called twice

def test_gives_up_after_5_attempts(respx_mock):
    # always 503 -> pytest.raises(httpx.HTTPStatusError)
```

- [ ] **Step 2:** `uv run pytest tests/test_github_api.py -v` → FAIL.
- [ ] **Step 3:** Implement with headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`; pagination via `response.links["next"]`; tenacity retry on 429, 5xx, and 403 responses whose `X-RateLimit-Remaining` is `"0"` — exponential backoff with jitter, max 5 attempts, honoring `Retry-After` when present; `body: null` → `""`; `state` from the API, but PRs with `pull_request.merged_at` set get `state="merged"`.
- [ ] **Step 4:** `uv run pytest tests/test_github_api.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): GitHub issues/PRs client with retries"`

---

### Task 7: Docs source (shallow git clone)

**Files:**
- Create: `src/copilot/ingest/docs_repo.py`, `tests/test_docs_repo.py`

**Interfaces:**
- Produces: `iter_docs(repo: str, workdir: Path, clone_url: str | None = None, branch: str | None = None) -> Iterator[SourceDoc]` — `git clone --depth 1` (or `git pull` if already cloned) into `workdir/<owner>__<name>`; yields one `SourceDoc(doc_type="doc", source_id="doc:<relpath>", title=<first H1 or filename>, url="https://github.com/<repo>/blob/<branch>/<relpath>", updated_at=<HEAD commit time>, posts=[])` per `*.md`/`*.mdx` file, skipping paths containing `node_modules`, `.github`, `vendor`, or a `CHANGELOG`-prefixed filename only if over 200 KB. `clone_url` defaults to `https://github.com/<repo>.git`; `branch` defaults to the clone's current branch.

- [ ] **Step 1: Write the failing test** — fixture creates a local git repo in `tmp_path` with `README.md` (`# Project`), `docs/guide.md` (`# Guide`), `node_modules/x.md`, `src/a.py`; commits.

```python
def test_iter_docs_from_local_repo(local_repo, tmp_path):
    docs = list(iter_docs("o/r", tmp_path / "work", clone_url=f"file://{local_repo}"))
    assert sorted(d.source_id for d in docs) == ["doc:README.md", "doc:docs/guide.md"]
    guide = next(d for d in docs if d.source_id == "doc:docs/guide.md")
    assert guide.title == "Guide" and guide.url.endswith("/docs/guide.md")

def test_second_run_pulls_instead_of_cloning(local_repo, tmp_path): ...  # run twice, no error, same results
```

- [ ] **Step 2:** `uv run pytest tests/test_docs_repo.py -v` → FAIL.
- [ ] **Step 3:** Implement using `subprocess.run([...], check=True)` (no shell).
- [ ] **Step 4:** `uv run pytest tests/test_docs_repo.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): markdown docs from shallow clone"`

---

### Task 8: Ingestion pipeline and incremental sync

**Files:**
- Create: `src/copilot/ingest/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Embedder` (Task 3), `redact` (Task 4), `chunk_markdown` / `chunk_thread` / `SourceDoc` (Task 5), `GitHubClient.iter_threads` (Task 6), `iter_docs` (Task 7).
- Produces:
  - `@dataclass IngestStats: seen: int = 0; skipped_unchanged: int = 0; upserted: int = 0; chunks_written: int = 0`
  - `ingest_documents(conn, embedder: Embedder, docs: Iterable[SourceDoc]) -> IngestStats` — per document, in its **own transaction**: redact `title`, `body`, every `post.body`; compute `content_hash` (sha256 of redacted title+body+posts); if a row with the same `(repo, source_id, content_hash)` exists → skip; else upsert `documents`, delete its chunks, chunk (`chunk_markdown` for docs, `chunk_thread` for issue/pr), embed, insert chunks (copying `doc_type`).
  - `sync_repo(conn, embedder: Embedder, gh: GitHubClient, repo: str, since_months: int, workdir: Path) -> dict[str, IngestStats]` — keys `"docs"` and `"threads"`; threads `since` = `sync_state.last_synced_at` for `(repo, "threads")` or `now - since_months`; records `started_at = now()` **before** fetching and writes it to `sync_state` **only after** the source finishes without error.

- [ ] **Step 1: Write the failing tests** (use `FakeEmbedder`; `SourceDoc`s built inline)

```python
def test_ingest_writes_documents_and_chunks(db): ...  # 1 doc + 1 issue -> 2 documents rows, chunks>0, embeddings 1536-dim

def test_reingest_unchanged_is_noop(db):
    emb = FakeEmbedder(); ingest_documents(db, emb, docs); calls = emb.calls
    stats = ingest_documents(db, emb, docs)
    assert stats.skipped_unchanged == len(docs) and emb.calls == calls
    assert db.execute("select count(*) n from chunks").fetchone()["n"] == first_count

def test_changed_doc_replaces_chunks(db): ...  # edit body -> old chunk text gone, no duplicate chunk_index

def test_pii_never_stored(db):
    ingest_documents(db, FakeEmbedder(), [issue_with_body("contact jane.doe@example.com")])
    text = " ".join(r["content"] for r in db.execute("select content from chunks"))
    assert "jane.doe@example.com" not in text and "<EMAIL_ADDRESS>" in text

def test_failure_midway_keeps_progress_and_does_not_advance_sync_state(db, tmp_path):
    # FakeGitHub yields 2 docs then raises httpx.HTTPStatusError; iter_docs monkeypatched to []
    with pytest.raises(httpx.HTTPStatusError):
        sync_repo(db, FakeEmbedder(), fake_gh, "o/r", 12, tmp_path)
    assert db.execute("select count(*) n from documents").fetchone()["n"] == 2
    assert db.execute("select * from sync_state where source='threads'").fetchone() is None

def test_second_sync_uses_last_synced_at(db, tmp_path): ...  # fake_gh records `since` arg; 2nd call since == 1st started_at
```

- [ ] **Step 2:** `uv run pytest tests/test_pipeline.py -v` → FAIL.
- [ ] **Step 3:** Implement as specified. `BudgetExceeded` propagates exactly like the HTTP error (no special handling needed — covered by the same path).
- [ ] **Step 4:** `uv run pytest tests/test_pipeline.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(ingest): idempotent pipeline with incremental sync"`

---

### Task 9: Dense, sparse, and hybrid (RRF) search

**Files:**
- Create: `src/copilot/retrieval/__init__.py`, `src/copilot/retrieval/models.py`, `src/copilot/retrieval/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `Embedder` (Task 3); `chunks`/`documents` tables (Task 2).
- Produces:
  - `@dataclass(frozen=True) SearchFilters: doc_types: tuple[str, ...] | None = None; state: str | None = None; updated_after: datetime | None = None`
  - `@dataclass Hit: chunk_id: int; document_id: int; content: str; score: float; doc_type: str; title: str; url: str; number: int | None`
  - `dense_search(conn, query_vec: list[float], k: int, filters: SearchFilters | None = None) -> list[Hit]` (score = `1 - cosine distance`)
  - `sparse_search(conn, query: str, k: int, filters: SearchFilters | None = None) -> list[Hit]` (score = `paradedb.score(c.id)`; predicate `c.id @@@ paradedb.match('content', %s)`)
  - `rrf_fuse(rankings: list[list[Hit]], k: int = 60, limit: int = 10) -> list[Hit]` (score = Σ 1/(k + rank), rank 1-based; ties keep first-seen order)
  - `SearchMode = Literal["dense", "sparse", "hybrid"]`; `search(conn, embedder: Embedder, query: str, mode: SearchMode = "hybrid", k: int = 10, filters: SearchFilters | None = None) -> list[Hit]` — hybrid fetches `k * 3` from each side, then fuses to `k`.
- Plan 2 adds reranking and query decomposition on top of `search()`; plan 4's retrieval experiment calls `search()` with each mode.

- [ ] **Step 1: Write the failing tests**

```python
def test_rrf_scores():
    a, b, c = hit(1), hit(2), hit(3)
    fused = rrf_fuse([[a, b], [b, c]], k=60, limit=3)
    assert [h.chunk_id for h in fused] == [2, 1, 3]
    assert fused[0].score == pytest.approx(1/62 + 1/61)

def test_sparse_finds_exact_identifier(seeded_db):
    assert sparse_search(seeded_db, "ERR_TIMEOUT", k=3)[0].content.startswith("ERR_TIMEOUT")

def test_punctuation_query_does_not_error(seeded_db):
    for mode in ("dense", "sparse", "hybrid"):
        search(seeded_db, FakeEmbedder(), 'why? (JWT) "auth" AND: #4521', mode=mode, k=3)

def test_filters_restrict_doc_type_and_state(seeded_db):
    hits = search(seeded_db, FakeEmbedder(), "background tasks", k=10,
                  filters=SearchFilters(doc_types=("issue",), state="open"))
    assert hits and all(h.doc_type == "issue" for h in hits)

def test_hybrid_contains_best_of_both(seeded_db):
    # a query whose best sparse hit and best dense hit differ -> both appear in hybrid top-k
```

(`seeded_db` — add to `tests/conftest.py` so Task 10 can reuse it — ingests ~6 hand-written `SourceDoc`s via `ingest_documents` with `FakeEmbedder`: at least one doc, one open issue, one closed issue, one PR; one chunk begins `"ERR_TIMEOUT"`.)

- [ ] **Step 2:** `uv run pytest tests/test_search.py -v` → FAIL.
- [ ] **Step 3:** Implement. Both SQL queries join `documents d ON d.id = c.document_id` and apply filters as parameterized `WHERE` clauses (`c.doc_type = ANY(%s)`, `d.state = %s`, `d.updated_at >= %s`) — never string-formatted.
- [ ] **Step 4:** `uv run pytest tests/test_search.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(retrieval): dense, BM25, and RRF hybrid search"`

---

### Task 10: CLI and first real index

**Files:**
- Create: `src/copilot/cli.py`, `tests/test_cli.py`, `README.md`
- Modify: `pyproject.toml` (`[project.scripts] copilot = "copilot.cli:app"`)

**Interfaces:**
- Consumes: everything above.
- Produces: Typer `app` with commands:
  - `copilot init-db` → `apply_schema`
  - `copilot sync [--repo TEXT] [--since-months INT]` (defaults from settings) → prints per-source `IngestStats` and `spent_usd()` after the run
  - `copilot search QUERY [--mode dense|sparse|hybrid] [-k INT] [--type doc|issue|pr]...` → numbered results: `score  [type #number]  title  url` + first 200 chars
  - `copilot spend` → total spend vs. ceiling
  - module-level `build_embedder(conn, settings) -> Embedder` (returns `LLMGateway`), patched in tests.

- [ ] **Step 1: Write the failing test** — `CliRunner` against `seeded_db` with `build_embedder` monkeypatched to `FakeEmbedder` and `COPILOT_DATABASE_URL` pointed at the test DB:

```python
def test_search_command_prints_results(...):
    r = runner.invoke(app, ["search", "ERR_TIMEOUT", "--mode", "sparse", "-k", "2"])
    assert r.exit_code == 0 and "ERR_TIMEOUT" in r.output and "[issue #" in r.output

def test_spend_command(...):
    r = runner.invoke(app, ["spend"]); assert "of $8.00" in r.output
```

- [ ] **Step 2:** `uv run pytest tests/test_cli.py -v` → FAIL.
- [ ] **Step 3:** Implement the CLI.
- [ ] **Step 4:** `uv run pytest -v` → **all** tests PASS.
- [ ] **Step 5: Pick the demo repo with the author.** Propose 3 mid-sized, active repos with real design discussion in issues/PRs and roughly ≤3,000 issues+PRs updated in the last 12 months (check with `gh api "search/issues?q=repo:OWNER/NAME+updated:>=YYYY-MM-DD" --jq .total_count`). **Ask the author to choose**; set `COPILOT_DEMO_REPO` in `.env`.
- [ ] **Step 6: First real sync.** Copy `.env.example` → `.env` with real keys (GitHub fine-grained PAT, read-only public repos). Run `uv run copilot init-db && uv run copilot sync`. Expected: completes; `copilot spend` shows well under **$0.50**. Re-run `uv run copilot sync` → `upserted: 0` for docs and only recently-updated threads, near-zero spend.
- [ ] **Step 7: Smoke-check quality by hand.** Run 3 real questions in each mode, e.g. `uv run copilot search "<an exact error string from an issue>" --mode dense` vs `--mode sparse` vs `--mode hybrid`. Note in `README.md` one example where sparse beats dense and one where dense beats sparse — this is the seed of the plan-4 retrieval experiment.
- [ ] **Step 8:** `README.md`: what the project is (one paragraph from spec §1), quickstart (`docker compose up -d db`, `.env`, `uv sync`, `copilot init-db`, `copilot sync`, `copilot search`), and the budget note.
- [ ] **Step 9: Commit** — `git add -A && git commit -m "feat(cli): init-db, sync, search, spend; first real index"`

---

## Done when

- `uv run pytest -v` is green with the DB container up.
- A real index of the chosen demo repo exists; `copilot search` returns sensible results in all three modes.
- Total spend so far is under $0.50 and visible via `copilot spend`.
