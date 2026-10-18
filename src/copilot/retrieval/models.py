from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SearchFilters:
    doc_types: tuple[str, ...] | None = None
    state: str | None = None
    updated_after: datetime | None = None


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    content: str
    score: float
    doc_type: str
    title: str
    url: str
    number: int | None
