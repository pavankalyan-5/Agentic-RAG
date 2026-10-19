"""Cross-encoder reranking: score (query, chunk) pairs jointly, keep the best few."""

from dataclasses import replace
from typing import Protocol

from copilot.retrieval.models import Hit


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """fastembed ONNX cross-encoder: runs on CPU, no PyTorch. The model loads on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # heavy import, only when needed

            self._model = TextCrossEncoder(self.model_name)
        return [float(s) for s in self._model.rerank(query, texts)]


def rerank(
    reranker: Reranker, query: str, hits: list[Hit], keep: int, min_score: float | None = None
) -> list[Hit]:
    if not hits:
        return []
    scores = reranker.score(query, [h.content for h in hits])
    ranked = sorted(zip(scores, range(len(hits))), key=lambda pair: pair[0], reverse=True)
    kept = [replace(hits[i], score=s) for s, i in ranked if min_score is None or s >= min_score]
    return kept[:keep]
