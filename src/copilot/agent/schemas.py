"""Structured outputs the LLM must return. Field descriptions are sent to the model as instructions."""

from typing import Literal

from pydantic import BaseModel, Field

from copilot.retrieval.agentic import SubQuery


class LiveLookup(BaseModel):
    kind: Literal["issue", "pull_request"]
    number: int = Field(description="The issue or pull request number, e.g. 3564.")


class RequestedAction(BaseModel):
    kind: Literal["comment_on_issue", "create_issue"]
    issue_number: int | None = Field(description="Issue/PR to comment on; null for create_issue.")
    instructions: str = Field(description="What the user wants the comment/issue to say, in their words.")


class ActionDraft(BaseModel):
    title: str | None = Field(description="Issue title for create_issue; null for comments.")
    body: str = Field(description="The comment/issue body in Markdown.")


class Plan(BaseModel):
    needs_search: bool = Field(
        description="Whether to search the indexed docs/issues/PRs. False for small talk, and for questions "
        "answered entirely by live_lookups (e.g. 'is #123 merged?')."
    )
    sub_queries: list[SubQuery] = Field(
        description="1-3 focused, self-contained searches covering every part of the question. "
        "Empty when needs_search is false."
    )
    live_lookups: list[LiveLookup] = Field(
        default_factory=list,
        description="Up to 3 specific numbered issues/PRs whose CURRENT status the user asks about "
        "(open/closed/merged, author, latest activity). Empty otherwise.",
    )
    requested_action: RequestedAction | None = Field(
        default=None,
        description="Only when the user EXPLICITLY asks you to post a comment or open an issue. "
        "A human must approve it before anything is posted. Null otherwise.",
    )


class AnswerDraft(BaseModel):
    answer: str = Field(
        description="Markdown answer. Put [chunk_id] right after every sentence that states a fact from the context."
    )
    cited_chunk_ids: list[int] = Field(description="Every chunk_id cited in the answer.")
    insufficient_context: bool = Field(
        description="True if the context does not contain the answer. Then say so plainly instead of guessing."
    )
