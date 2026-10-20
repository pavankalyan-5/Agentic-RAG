"""LLM-drafted eval cases, for HUMAN REVIEW (they become golden only after a person checks them).

Each draft is anchored to the document(s) it was written from, so its expected sources are known.
Unanswerable and adversarial cases are hand-written (evals/seed_cases.jsonl): LLMs make them too easy.
"""

import random

import psycopg
from pydantic import BaseModel, Field

from copilot.evals.dataset import EvalCase
from copilot.llm.gateway import ChatModel

DRAFT_SYSTEM = """\
You write evaluation questions for a Q&A assistant that answers from a software project's docs, issues \
and pull requests.

Given the document(s) below, write ONE question a developer would realistically ask that is answered by \
the document(s), plus a short reference answer that uses ONLY facts in them.
- Use your own words: do NOT copy headings, titles, or distinctive phrases from the document (that would \
make keyword search trivially easy and the evaluation meaningless).
- The question must make sense on its own (no "this document", "the above").
- If two documents are given, the question must need BOTH to be answered fully.
- If the document has no substantive answerable content (boilerplate, license, link lists), set \
answerable to false.
"""

MAX_DOC_CHARS = 6000
CI_PER_CATEGORY = 3  # suggested PR-subset size per category; the reviewer adjusts it


class DraftCase(BaseModel):
    question: str = Field(description="A natural developer question, in your own words.")
    reference_answer: str = Field(description="A short correct answer using only facts from the document(s).")
    answerable: bool = Field(description="False if the document has nothing substantive to ask about.")


def _documents(conn: psycopg.Connection, repo: str, min_chars: int) -> list[dict]:
    rows = conn.execute(
        """
        select d.source_id, d.doc_type, d.title, d.number,
               string_agg(c.content, E'\\n\\n' order by c.chunk_index) as text
        from documents d join chunks c on c.document_id = d.id
        where d.repo = %s group by d.id order by d.source_id
        """,
        (repo,),
    ).fetchall()
    return [dict(r) for r in rows if len(r["text"]) >= min_chars]


def _render(doc: dict) -> str:
    return f"Source: {doc['source_id']}\n<document>\n{doc['text'][:MAX_DOC_CHARS]}\n</document>"


def draft_cases(
    conn: psycopg.Connection,
    chat: ChatModel,
    *,
    repo: str,
    per_category: dict[str, int],
    seed: int = 42,
    model: str = "gpt-4.1-mini",
    min_chars: int = 0,
) -> list[EvalCase]:
    rng = random.Random(seed)  # same seed -> same sample -> reproducible drafts
    docs = _documents(conn, repo, min_chars)
    pages = [d for d in docs if d["doc_type"] == "doc"]
    threads = [d for d in docs if d["doc_type"] in ("issue", "pr")]
    prefix = {"docs": "docs", "history": "hist", "multi_part": "multi"}
    out: list[EvalCase] = []

    def draft(category: str, sources: list[dict]) -> None:
        title = " | ".join(f"{d['title']}" + (f" (#{d['number']})" if d["number"] else "") for d in sources)
        user = f"Title: {title}\n\n" + "\n\n".join(_render(d) for d in sources)
        result = chat.chat([{"role": "system", "content": DRAFT_SYSTEM}, {"role": "user", "content": user}],
                           model=model, schema=DraftCase, operation="eval_draft")
        if not result.answerable or not result.reference_answer.strip():
            return
        n = sum(1 for c in out if c.category == category) + 1
        out.append(EvalCase(id=f"{prefix[category]}-{n:02d}", category=category, question=result.question,
                            expected_sources=[d["source_id"] for d in sources],
                            reference_answer=result.reference_answer, in_ci=n <= CI_PER_CATEGORY,
                            notes="LLM draft: review wording, answer, and sources before promoting to golden"))

    for doc in rng.sample(pages, min(per_category.get("docs", 0), len(pages))):
        draft("docs", [doc])
    for doc in rng.sample(threads, min(per_category.get("history", 0), len(threads))):
        draft("history", [doc])
    # Two-source questions need RELATED documents: pair each anchor with its nearest neighbour.
    by_id = {d["source_id"]: d for d in pages + threads}
    pairs: set[frozenset] = set()
    for anchor in rng.sample(sorted(by_id), len(by_id)):
        if len(pairs) >= per_category.get("multi_part", 0):
            break
        partner = nearest_document(conn, repo, anchor)
        pair = frozenset((anchor, partner))
        if partner in by_id and pair not in pairs:
            pairs.add(pair)
            draft("multi_part", [by_id[anchor], by_id[partner]])
    return out


NEAR_DUPLICATE_DISTANCE = 0.02  # cosine distance below this = the same text (e.g. README vs docs/index)


def nearest_document(conn: psycopg.Connection, repo: str, source_id: str) -> str | None:
    """The most similar OTHER document, by the mean of each document's chunk embeddings (pgvector avg)."""
    row = conn.execute(
        """
        with doc_vecs as (
            select d.source_id, avg(c.embedding) as v
            from documents d join chunks c on c.document_id = d.id
            where d.repo = %(repo)s group by d.source_id
        )
        select b.source_id from doc_vecs a, doc_vecs b
        where a.source_id = %(sid)s and b.source_id <> a.source_id and (a.v <=> b.v) > %(dup)s
        order by a.v <=> b.v limit 1
        """,
        {"repo": repo, "sid": source_id, "dup": NEAR_DUPLICATE_DISTANCE},
    ).fetchone()
    return row["source_id"] if row else None
