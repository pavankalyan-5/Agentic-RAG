"""Structured outputs the LLM must return. Field descriptions are sent to the model as instructions."""

from pydantic import BaseModel, Field

from copilot.retrieval.agentic import SubQuery


class Plan(BaseModel):
    needs_search: bool = Field(
        description="False only for small talk (greetings, thanks) that needs no project knowledge."
    )
    sub_queries: list[SubQuery] = Field(
        description="1-3 focused, self-contained searches covering every part of the question. "
        "Empty when needs_search is false."
    )


class AnswerDraft(BaseModel):
    answer: str = Field(
        description="Markdown answer. Put [chunk_id] right after every sentence that states a fact from the context."
    )
    cited_chunk_ids: list[int] = Field(description="Every chunk_id cited in the answer.")
    insufficient_context: bool = Field(
        description="True if the context does not contain the answer. Then say so plainly instead of guessing."
    )
