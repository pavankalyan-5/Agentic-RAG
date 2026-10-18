"""Type-aware chunking: docs split at headings, issue/PR threads kept together where possible."""

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

from copilot.ingest.models import SourceDoc

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
PREFIX_MAX_TOKENS = 64  # headings/titles repeated on every chunk are capped so they can't eat the budget


@dataclass
class Chunk:
    index: int
    text: str
    heading: str | None


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def chunk_markdown(text: str, max_tokens: int = 500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        prefix = f"{_truncate(heading, _prefix_cap(max_tokens))}\n\n" if heading else ""
        for piece in _pack(prefix, _paragraphs(body), max_tokens):
            chunks.append(Chunk(index=len(chunks), text=piece, heading=heading))
    return chunks


def chunk_thread(doc: SourceDoc, max_tokens: int = 800) -> list[Chunk]:
    header = _truncate(f"#{doc.number} {doc.title}", _prefix_cap(max_tokens))
    parts = [p for p in [doc.body.strip()] + [f"{p.author or 'unknown'}: {p.body.strip()}" for p in doc.posts] if p]
    pieces = _pack(f"{header}\n\n", parts, max_tokens) or [header]
    return [Chunk(index=i, text=t, heading=None) for i, t in enumerate(pieces)]


def _prefix_cap(max_tokens: int) -> int:
    return min(PREFIX_MAX_TOKENS, max_tokens // 4)


def _truncate(text: str, max_tokens: int) -> str:
    tokens = _encoding().encode(text, disallowed_special=())
    return text if len(tokens) <= max_tokens else _encoding().decode(tokens[:max_tokens]).rstrip() + "…"


def markdown_title(text: str) -> str | None:
    """First level-1 heading outside fenced code blocks."""
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := _HEADING.match(line)) and len(m.group(1)) == 1:
            return m.group(2)
    return None


def _sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown at #..### headings, ignoring '#' lines inside fenced code blocks."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            sections.append((m.group(2), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(lines)) for h, lines in sections if h or "\n".join(lines).strip()]


def _paragraphs(body: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _pack(prefix: str, parts: list[str], max_tokens: int) -> list[str]:
    """Greedily pack parts into chunks of at most max_tokens, each starting with prefix."""
    budget = max_tokens - count_tokens(prefix)
    if budget <= 0:
        raise ValueError("prefix alone exceeds max_tokens")
    pieces: list[str] = []
    current: list[str] = []
    for part in parts:
        for sub in _hard_split(part, budget) if count_tokens(part) > budget else [part]:
            if current and count_tokens("\n\n".join(current + [sub])) > budget:
                pieces.append("\n\n".join(current))
                current = []
            current.append(sub)
    if current:
        pieces.append("\n\n".join(current))
    return [prefix + p for p in pieces]


def _hard_split(text: str, budget: int) -> list[str]:
    """Split by token count; re-check since decode/encode round-trips can shift counts slightly."""
    tokens = _encoding().encode(text, disallowed_special=())
    out: list[str] = []
    step = budget
    i = 0
    while i < len(tokens):
        piece = _encoding().decode(tokens[i : i + step])
        if count_tokens(piece) > budget and step > 1:
            step -= 1
            continue
        out.append(piece.strip())
        i += step
        step = budget
    return [p for p in out if p]
