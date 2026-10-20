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
from copilot.observability.tracing import set_attributes, span

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
# The API rejected the request itself (e.g. a schema strict mode can't accept): retrying won't help.
RequestRejected = openai.BadRequestError


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
        self._client = client

    @property
    def client(self):
        # Created on the first cache miss, not here: with no API key (CI on fork PRs), cached
        # embeddings/answers must still work, and only a real miss should fail (with OpenAI's error).
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def spent_usd(self) -> float:
        return spent_usd(self.conn)

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.settings.embedding_model
        with span("llm.embed", "EMBEDDING", embedding__model_name=model, copilot__operation="embed",
                  copilot__texts=len(texts)) as s:
            keys = [_cache_key(model, t) for t in texts]
            cached = self._cache_get(keys)
            if cached:
                self._record(model, "embed", 0, 0, 0.0, cached=True)

            missing = [(k, t) for k, t in zip(keys, texts) if k not in cached]
            tokens_total, cost_total = 0, 0.0
            for start in range(0, len(missing), EMBED_BATCH_SIZE):
                batch = missing[start : start + EMBED_BATCH_SIZE]
                self._check_budget()
                resp = self._create_embeddings(model, [t for _, t in batch])
                tokens = resp.usage.prompt_tokens
                for (key, _), item in zip(batch, resp.data):
                    cached[key] = item.embedding
                    self._cache_put(key, model, item.embedding)
                cost = cost_usd(model, tokens)
                self._record(model, "embed", tokens, 0, cost, cached=False)
                tokens_total, cost_total = tokens_total + tokens, cost_total + cost
            set_attributes(s, copilot__cache_hit=not missing, copilot__cost_usd=cost_total,
                           llm__token_count__prompt=tokens_total, llm__token_count__total=tokens_total)
            return [cached[k] for k in keys]

    def chat(self, messages: list[dict], *, model: str, schema: type[T], operation: str) -> T:
        """Structured-output chat: returns an instance of `schema`. Identical requests are free."""
        # The gateway is the one LLM span per call: it also sees cache hits, which never reach OpenAI.
        with span(f"llm.{operation}", "LLM", llm__model_name=model, llm__provider="openai",
                  copilot__operation=operation, input__value=messages[-1]["content"][:4000]) as s:
            result, tokens = self._chat(messages, model=model, schema=schema, operation=operation)
            cost = cost_usd(model, *tokens) if tokens else 0.0
            set_attributes(s, copilot__cache_hit=tokens is None, copilot__cost_usd=cost,
                           output__value=result.model_dump_json()[:4000])
            if tokens:
                set_attributes(s, llm__token_count__prompt=tokens[0], llm__token_count__completion=tokens[1],
                               llm__token_count__total=tokens[0] + tokens[1])
            return result

    def _chat(self, messages: list[dict], *, model: str, schema: type[T], operation: str) -> tuple[T, tuple | None]:
        """Returns (result, (prompt_tokens, completion_tokens)) or (result, None) on a cache hit."""
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
            return schema.model_validate(hit), None

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
        return message.parsed, (usage.prompt_tokens, usage.completion_tokens)

    def complete(self, messages: list[dict], *, model: str, operation: str) -> str:
        """Free-text chat (for eval judges that want prose). Same cache, ledger and ceiling as chat()."""
        with span(f"llm.{operation}", "LLM", llm__model_name=model, llm__provider="openai",
                  copilot__operation=operation, input__value=messages[-1]["content"][:4000]) as s:
            key = _cache_key(model, json.dumps({"mode": "complete", "messages": messages}, sort_keys=True))
            hit = self._cache_get([key]).get(key)
            if hit is not None:
                self._record(model, operation, 0, 0, 0.0, cached=True)
                set_attributes(s, copilot__cache_hit=True, copilot__cost_usd=0.0, output__value=hit[:4000])
                return hit
            self._check_budget()
            resp = self._create_chat(model, messages)
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            cost = cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
            self._cache_put(key, model, text)
            self._record(model, operation, usage.prompt_tokens, usage.completion_tokens, cost, cached=False)
            set_attributes(s, copilot__cache_hit=False, copilot__cost_usd=cost, output__value=text[:4000],
                           llm__token_count__prompt=usage.prompt_tokens,
                           llm__token_count__completion=usage.completion_tokens,
                           llm__token_count__total=usage.prompt_tokens + usage.completion_tokens)
            return text

    @_retry_transient
    def _create_chat(self, model: str, messages: list[dict]):
        return self.client.chat.completions.create(model=model, messages=messages, temperature=0)

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
