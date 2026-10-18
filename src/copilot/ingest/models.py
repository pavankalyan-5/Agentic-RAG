from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

DocType = Literal["doc", "issue", "pr"]


@dataclass
class Post:
    author: str | None
    body: str
    created_at: datetime | None


@dataclass
class SourceDoc:
    repo: str
    source_id: str  # "doc:<path>" | "issue:<n>" | "pr:<n>"
    doc_type: DocType
    number: int | None
    title: str
    url: str
    author: str | None
    labels: list[str]
    state: str | None
    created_at: datetime | None
    updated_at: datetime
    body: str
    posts: list[Post] = field(default_factory=list)
