"""Cross-encoder reranking: score (query, chunk) pairs jointly, keep the best few."""

from dataclasses import replace
from typing import Protocol

from copilot.observability.tracing import record_documents, span
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
    reranker: Reranker, query: str, hits: list[Hit], keep: int, min_score: float | None = None,
    max_per_document: int | None = None,
) -> list[Hit]:
    """max_per_document: several chunks of one long PR could fill every slot and crowd out the
    document that answers the question (eval finding, 2026-10-20)."""
    if not hits:
        return []
    with span("rerank", "RERANKER", input__value=query, reranker__top_k=keep, copilot__candidates=len(hits),
              copilot__min_score=min_score) as s:
        scores = reranker.score(query, [h.content for h in hits])
        ranked = sorted(zip(scores, range(len(hits))), key=lambda pair: pair[0], reverse=True)
        kept, per_document = [], {}
        for sc, i in ranked:
            doc = hits[i].document_id
            if (min_score is not None and sc < min_score) or (
                    max_per_document and per_document.get(doc, 0) >= max_per_document):
                continue
            per_document[doc] = per_document.get(doc, 0) + 1
            kept.append(replace(hits[i], score=sc))
            if len(kept) == keep:
                break
        record_documents(s, kept, prefix="reranker.output_documents")
        return kept
