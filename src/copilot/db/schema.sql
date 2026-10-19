CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- One row per source item: a doc page, an issue, or a PR.
CREATE TABLE IF NOT EXISTS documents (
    id            bigserial PRIMARY KEY,
    repo          text NOT NULL,
    source_id     text NOT NULL,              -- "doc:<path>" | "issue:<n>" | "pr:<n>"
    doc_type      text NOT NULL CHECK (doc_type IN ('doc', 'issue', 'pr')),
    number        int,
    title         text NOT NULL,
    url           text NOT NULL,
    author        text,
    labels        text[] NOT NULL DEFAULT '{}',
    state         text,
    created_at    timestamptz,
    updated_at    timestamptz NOT NULL,
    content_hash  text NOT NULL,
    UNIQUE (repo, source_id)
);

-- Searchable pieces of documents. doc_type is copied here so BM25 can filter on it.
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  int NOT NULL,
    heading      text,
    content      text NOT NULL,
    doc_type     text NOT NULL,
    embedding    vector(1536) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks USING bm25 (id, content, doc_type);

-- Last successful sync per repo and source ("docs" | "threads").
CREATE TABLE IF NOT EXISTS sync_state (
    repo            text NOT NULL,
    source          text NOT NULL,
    last_synced_at  timestamptz NOT NULL,
    PRIMARY KEY (repo, source)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key         text PRIMARY KEY,
    model       text NOT NULL,
    response    jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Run status per conversation thread (the conversation itself lives in LangGraph's checkpoints).
-- lease_until lets a restarted server find runs that died mid-way, and claim each exactly once.
CREATE TABLE IF NOT EXISTS thread_metadata (
    thread_id    text PRIMARY KEY,
    status       text NOT NULL CHECK (status IN ('idle', 'running', 'awaiting_approval', 'done', 'failed')),
    lease_until  timestamptz,
    last_error   text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Added after the plan-3 review (ALTER so existing databases upgrade in place):
--   lease_owner:   fencing token of the worker holding the lease; status writes require it
--   pending_input: the question being answered, so recovery can replay one LangGraph never saw
--   owner:         fingerprint of the API key that created the thread (readers see only their own)
ALTER TABLE thread_metadata ADD COLUMN IF NOT EXISTS lease_owner text;
ALTER TABLE thread_metadata ADD COLUMN IF NOT EXISTS pending_input jsonb;
ALTER TABLE thread_metadata ADD COLUMN IF NOT EXISTS owner text;

-- Every approved/rejected GitHub action. The primary key makes execution idempotent:
-- a second approval of the same action finds the row and returns the stored result.
CREATE TABLE IF NOT EXISTS actions (
    idempotency_key  text PRIMARY KEY,
    thread_id        text NOT NULL,
    kind             text NOT NULL,
    target_repo      text NOT NULL,
    payload          jsonb NOT NULL,
    -- 'pending' = claimed by the executor before calling GitHub, so a concurrent approval can't also run it
    status           text NOT NULL CHECK (status IN ('pending', 'executed', 'rejected', 'failed')),
    result           jsonb,
    decided_by       text,
    decided_at       timestamptz NOT NULL DEFAULT now()
);

-- 'unknown' = the GitHub call's outcome is uncertain (timeout/crash mid-call); reconciled later.
ALTER TABLE actions DROP CONSTRAINT IF EXISTS actions_status_check;
ALTER TABLE actions ADD CONSTRAINT actions_status_check
    CHECK (status IN ('pending', 'executed', 'rejected', 'failed', 'unknown'));

-- Every OpenAI call (and cache hit) — the source of truth for the budget ceiling.
CREATE TABLE IF NOT EXISTS llm_usage (
    id             bigserial PRIMARY KEY,
    at             timestamptz NOT NULL DEFAULT now(),
    model          text NOT NULL,
    operation      text NOT NULL,
    input_tokens   int NOT NULL,
    output_tokens  int NOT NULL DEFAULT 0,
    cost_usd       numeric(12, 6) NOT NULL,
    cached         bool NOT NULL
);
