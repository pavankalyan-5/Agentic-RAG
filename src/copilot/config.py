from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://copilot:copilot@localhost:5433/copilot"
    openai_api_key: str = ""
    github_token: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    budget_usd: float = 8.0
    demo_repo: str = ""
    since_months: int = 12
    # Agent (plan 2). Non-reasoning models: no hidden reasoning tokens, so cost is predictable.
    planner_model: str = "gpt-4.1-nano"
    answer_model: str = "gpt-4.1-mini"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_keep: int = 4
    # Calibrated 2026-10-19 on real Starlette questions: relevant chunks scored >= -0.5, irrelevant
    # ones <= -1.14 (docs/results/2026-10-19-findings.md). Provisional until plan-4 evals.
    rerank_min_score: float | None = 3.0
    # One chunk per document per sub-query: chunks of one closed PR crowded out the merged fix (eval
    # sweep 2026-10-20: merged fix in context 3/6 -> 5/6, hit rate unchanged). Bonus for merged PRs didn't help.
    rerank_max_per_document: int | None = 1
    # Live tools & actions (plan 3). Writes only ever target sandbox_repo, with a token scoped to it.
    sandbox_repo: str = ""
    github_write_token: str = ""
    mcp_image: str = "ghcr.io/github/github-mcp-server:v2.0.2"
    mcp_timeout_s: float = 15.0
    # JSON in env, e.g. COPILOT_API_KEYS='{"k1": "maintainer", "k2": "reader"}'
    api_keys: dict[str, Literal["reader", "maintainer"]] = {}
    # Observability (plan 4): tracing is off unless an OTLP endpoint is set, e.g.
    # COPILOT_PHOENIX_ENDPOINT=http://127.0.0.1:6006/v1/traces
    phoenix_endpoint: str = ""
    phoenix_project: str = "dev-docs-copilot"
    # Evals (plan 4): the LLM that grades answers. Goes through the gateway like every other call.
    eval_judge_model: str = "gpt-4.1-mini"
    # A running turn's heartbeat renews its lease; a crashed one is resumed once the lease expires.
    run_lease_s: float = 60.0
    recover_interval_s: float = 15.0  # how often the server looks for expired leases


@lru_cache
def get_settings() -> Settings:
    return Settings()
