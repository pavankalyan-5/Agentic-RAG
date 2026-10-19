"""The only module that talks to OpenAI: caches responses, records cost, enforces the budget."""

import hashlib
import json
from typing import Protocol, TypeVar

import openai
import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from copilot.config import Settings
from copilot.llm.pricing import cost_usd

EMBED_BATCH_SIZE = 96

T = TypeVar("T", bound=BaseModel)

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


# Base class of every OpenAI SDK error (auth, connection, rate limit after retries, ...), re-exported
# so callers can handle provider failures without importing openai themselves.
ProviderError = openai.OpenAIError


class LLMOutputError(Exception):
    """The model refused, or its structured output couldn't be parsed. Safe to retry."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def chat(self, messages: list[dict], *, model: str, schema: type[T], operation: str) -> T: ...


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
            self._record(model, "embed", 0, 0, 0.0, cached=True)

        missing = [(k, t) for k, t in zip(keys, texts) if k not in cached]
        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start : start + EMBED_BATCH_SIZE]
            self._check_budget()
            resp = self._create_embeddings(model, [t for _, t in batch])
            tokens = resp.usage.prompt_tokens
            for (key, _), item in zip(batch, resp.data):
                cached[key] = item.embedding
                self._cache_put(key, model, item.embedding)
            self._record(model, "embed", tokens, 0, cost_usd(model, tokens), cached=False)

        return [cached[k] for k in keys]

    def chat(self, messages: list[dict], *, model: str, schema: type[T], operation: str) -> T:
        """Structured-output chat: returns an instance of `schema`. Identical requests are free."""
        key = _cache_key(
            model,
            json.dumps(
                {"schema": schema.__name__, "json_schema": schema.model_json_schema(), "messages": messages},
                sort_keys=True,
            ),
        )
        hit = self._cache_get([key]).get(key)
        if hit is not None:
            self._record(model, operation, 0, 0, 0.0, cached=True)
            return schema.model_validate(hit)

        self._check_budget()
        resp = self._parse(model, messages, schema)
        message = resp.choices[0].message
        if message.refusal or message.parsed is None:
            raise LLMOutputError(f"{operation}: model returned no parsable {schema.__name__} ({message.refusal!r})")
        usage = resp.usage
        self._cache_put(key, model, message.parsed.model_dump())
        self._record(
            model, operation, usage.prompt_tokens, usage.completion_tokens,
            cost_usd(model, usage.prompt_tokens, usage.completion_tokens), cached=False,
        )
        return message.parsed

    @_retry_transient
    def _create_embeddings(self, model: str, inputs: list[str]):
        return self.client.embeddings.create(model=model, input=inputs)

    @_retry_transient
    def _parse(self, model: str, messages: list[dict], schema: type[BaseModel]):
        return self.client.chat.completions.parse(
            model=model, messages=messages, response_format=schema, temperature=0
        )

    def _check_budget(self) -> None:
        spent = self.spent_usd()
        if spent >= self.settings.budget_usd:
            raise BudgetExceeded(f"spent ${spent:.4f} of ${self.settings.budget_usd:.2f} ceiling")

    def _cache_get(self, keys: list[str]) -> dict:
        rows = self.conn.execute(
            "select key, response from llm_cache where key = any(%s)", (keys,)
        ).fetchall()
        return {r["key"]: r["response"] for r in rows}

    def _cache_put(self, key: str, model: str, response) -> None:
        self.conn.execute(
            "insert into llm_cache (key, model, response) values (%s, %s, %s) on conflict (key) do nothing",
            (key, model, Jsonb(response)),
        )

    def _record(
        self, model: str, operation: str, input_tokens: int, output_tokens: int, cost: float, cached: bool
    ) -> None:
        self.conn.execute(
            "insert into llm_usage (model, operation, input_tokens, output_tokens, cost_usd, cached)"
            " values (%s, %s, %s, %s, %s, %s)",
            (model, operation, input_tokens, output_tokens, cost, cached),
        )
        self.conn.commit()


def spent_usd(conn: psycopg.Connection) -> float:
    row = conn.execute("select coalesce(sum(cost_usd), 0) as total from llm_usage").fetchone()
    return float(row["total"])


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()
