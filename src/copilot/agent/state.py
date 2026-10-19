"""The agent's state. Everything here is checkpointed to Postgres, so it must be plain JSON-like data."""

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # {"role", "content"}; appended across turns
    question: str  # the current question
    plan: list[dict]  # SubQuery dicts from the planner
    retrieved_docs: list[dict]  # Hit dicts after reranking
    draft: dict | None  # AnswerDraft dict from the answer model
    next_step: str  # "retry" | "done", set by verify
    error_context: str | None  # feedback for self-correction
    correction_attempts: int
    answer: dict | None  # final {"text", "sources", "insufficient_context"}
