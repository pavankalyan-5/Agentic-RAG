"""The only module that talks to OpenAI: caches responses, records cost, enforces the budget."""

import hashlib
from typing import Protocol

import openai
import psycopg
from psycopg.types.json import Jsonb
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from copilot.config import Settings
from copilot.llm.pricing import cost_usd

EMBED_BATCH_SIZE = 96

_retry_transient = retry(
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
    ),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


class BudgetExceeded(Exception):
    pass


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMGateway:
    def __init__(self, conn: psycopg.Connection, settings: Settings, client=None):
        self.conn = conn
        self.settings = settings
        self.client = client or openai.OpenAI(api_key=settings.openai_api_key)

    def spent_usd(self) -> float:
        return spent_usd(self.conn)

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.settings.embedding_model
        keys = [_cache_key(model, t) for t in texts]
        cached = self._cache_get(keys)
        if cached:
            self._record(model, "embed", 0, 0.0, cached=True)

        missing = [(k, t) for k, t in zip(keys, texts) if k not in cached]
        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start : start + EMBED_BATCH_SIZE]
            self._check_budget()
            resp = self._create_embeddings(model, [t for _, t in batch])
            tokens = resp.usage.prompt_tokens
            for (key, _), item in zip(batch, resp.data):
                cached[key] = item.embedding
                self.conn.execute(
                    "insert into llm_cache (key, model, response) values (%s, %s, %s)"
                    " on conflict (key) do nothing",
                    (key, model, Jsonb(item.embedding)),
                )
            self._record(model, "embed", tokens, cost_usd(model, tokens), cached=False)

        return [cached[k] for k in keys]

    @_retry_transient
    def _create_embeddings(self, model: str, inputs: list[str]):
        return self.client.embeddings.create(model=model, input=inputs)

    def _check_budget(self) -> None:
        spent = self.spent_usd()
        if spent >= self.settings.budget_usd:
            raise BudgetExceeded(f"spent ${spent:.4f} of ${self.settings.budget_usd:.2f} ceiling")

    def _cache_get(self, keys: list[str]) -> dict[str, list[float]]:
        rows = self.conn.execute(
            "select key, response from llm_cache where key = any(%s)", (keys,)
        ).fetchall()
        return {r["key"]: r["response"] for r in rows}

    def _record(self, model: str, operation: str, tokens: int, cost: float, cached: bool) -> None:
        self.conn.execute(
            "insert into llm_usage (model, operation, input_tokens, cost_usd, cached)"
            " values (%s, %s, %s, %s, %s)",
            (model, operation, tokens, cost, cached),
        )
        self.conn.commit()


def spent_usd(conn: psycopg.Connection) -> float:
    row = conn.execute("select coalesce(sum(cost_usd), 0) as total from llm_usage").fetchone()
    return float(row["total"])


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()
