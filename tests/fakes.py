"""Test doubles for OpenAI. No network in tests."""

import hashlib
import math
import re
from types import SimpleNamespace

DIM = 1536


def hash_embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic bag-of-words vector: texts sharing words get similar vectors."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [hash_embed(t) for t in texts]


class FakeOpenAI:
    """Mimics the slice of openai.OpenAI that LLMGateway uses."""

    def __init__(self, chat_responses: list | None = None, usage: tuple[int, int] = (10, 5)):
        self.calls = 0
        self.last_input: list[str] | None = None
        self.embeddings = SimpleNamespace(create=self._create_embeddings)
        # Queue of pydantic instances to return from chat.completions.parse; None simulates a refusal.
        self.chat_responses = list(chat_responses or [])
        self.chat_usage = usage
        self.chat_calls = 0
        self.complete_responses: list = []  # texts for chat.completions.create (an Exception is raised)
        self.complete_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse, create=self._create))

    def _create(self, model: str, messages: list[dict], temperature: float = 0):
        self.complete_calls += 1
        text = self.complete_responses.pop(0)
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=None))],
            usage=SimpleNamespace(prompt_tokens=self.chat_usage[0], completion_tokens=self.chat_usage[1]),
        )

    def _parse(self, model: str, messages: list[dict], response_format, temperature: float = 0):
        self.chat_calls += 1
        parsed = self.chat_responses.pop(0)
        if isinstance(parsed, Exception):
            raise parsed
        message = SimpleNamespace(parsed=parsed, refusal="I can't help with that" if parsed is None else None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=self.chat_usage[0], completion_tokens=self.chat_usage[1]),
        )

    def _create_embeddings(self, model: str, input: list[str]):
        self.calls += 1
        self.last_input = list(input)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=hash_embed(t)) for t in input],
            usage=SimpleNamespace(prompt_tokens=sum(len(t.split()) for t in input)),
        )


class FakeTools:
    """ToolClient stand-in mirroring GitHub MCP shapes. Even PR numbers are merged. fail=True -> ToolError."""

    def __init__(self, fail: bool = False, missing: set[int] = frozenset(), write_error: Exception | None = None):
        self.fail, self.missing, self.write_error = fail, set(missing), write_error
        self.calls: list[tuple[str, dict]] = []
        self.comments: dict[int, list[dict]] = {}  # issue -> comments posted (read back by get_comments)

    def tool_names(self) -> set[str]:
        return {"issue_read", "pull_request_read", "add_issue_comment", "issue_write"}

    def call(self, name: str, arguments: dict):
        from copilot.tools.mcp_client import ToolError

        self.calls.append((name, arguments))
        if self.fail:
            raise ToolError("tool server unavailable")
        if self.write_error and name in ("add_issue_comment", "issue_write"):
            raise self.write_error
        if name == "issue_read" and arguments.get("method") == "get_comments":
            return list(self.comments.get(arguments["issue_number"], []))
        number = arguments.get("pullNumber") or arguments.get("issue_number")
        if number in self.missing:
            raise ToolError(f"{name} returned an error: not found")
        owner, repo = arguments["owner"], arguments["repo"]
        if name == "pull_request_read":
            merged = number % 2 == 0
            return {"number": number, "title": f"PR {number}", "state": "closed", "merged": merged,
                    "merged_at": "2026-10-10T00:00:00Z" if merged else None,
                    "html_url": f"https://github.com/{owner}/{repo}/pull/{number}", "user": {"login": "Kludex"},
                    "labels": [], "comments": 1, "updated_at": "2026-10-18T10:00:00Z", "body": "Preserve trailers."}
        if name == "issue_read":
            return {"number": number, "title": f"Issue {number}", "state": "open",
                    "html_url": f"https://github.com/{owner}/{repo}/issues/{number}", "user": {"login": "alice"},
                    "labels": [{"name": "bug"}], "comments": 2, "updated_at": "2026-10-18T10:00:00Z",
                    "body": "Something breaks."}
        # Write tools: same shape as GitHub MCP v2.0.2 (MinimalResponse {"id", "url"})
        url = f"https://github.com/{owner}/{repo}/issues/{number or 99}#issuecomment-{len(self.calls)}"
        if name == "add_issue_comment":
            self.comments.setdefault(number, []).append({"body": arguments["body"], "html_url": url})
        return {"id": str(len(self.calls)), "url": url}


class FakeReranker:
    """Scores a text by how many distinct query words it contains."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        words = set(re.findall(r"\w+", query.lower()))
        return [float(len(words & set(re.findall(r"\w+", t.lower())))) for t in texts]


class FakeChat:
    """Implements the ChatModel protocol from scripted responses per operation.

    responses: {operation: [response, ...]} — each call pops the next one; an Exception instance
    is raised instead of returned (to simulate LLMOutputError, BudgetExceeded, ...); a callable is
    called with the messages (e.g. to cite chunk ids that only exist at runtime).
    """

    def __init__(self, responses: dict[str, list] | None = None):
        self.responses = {op: list(rs) for op, rs in (responses or {}).items()}
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], *, model: str, schema, operation: str):
        self.calls.append({"operation": operation, "model": model, "messages": messages})
        response = self.responses[operation].pop(0)
        if isinstance(response, Exception):
            raise response
        return response(messages) if callable(response) else response

    def calls_for(self, operation: str) -> list[dict]:
        return [c for c in self.calls if c["operation"] == operation]
