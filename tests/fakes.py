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

    def __init__(self):
        self.calls = 0
        self.last_input: list[str] | None = None
        self.embeddings = SimpleNamespace(create=self._create_embeddings)

    def _create_embeddings(self, model: str, input: list[str]):
        self.calls += 1
        self.last_input = list(input)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=hash_embed(t)) for t in input],
            usage=SimpleNamespace(prompt_tokens=sum(len(t.split()) for t in input)),
        )
