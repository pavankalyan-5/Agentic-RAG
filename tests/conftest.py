import os

import psycopg
import pytest

from copilot.config import Settings
from copilot.db.connection import apply_schema, connect

ADMIN_URL = os.environ.get("COPILOT_TEST_ADMIN_URL", "postgresql://copilot:copilot@localhost:5433/copilot")
TEST_DB = "copilot_test"
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
TABLES = "documents, chunks, sync_state, llm_cache, llm_usage, thread_metadata, actions"


@pytest.fixture(scope="session")
def _test_database():
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    with connect(TEST_URL) as conn:
        apply_schema(conn)
    return TEST_URL


@pytest.fixture
def db(_test_database):
    conn = connect(_test_database)
    conn.execute(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def settings(_test_database):
    return Settings(_env_file=None, database_url=_test_database)


def _seed_docs():
    from datetime import datetime, timezone

    from copilot.ingest.models import Post, SourceDoc

    now = datetime(2026, 3, 1, tzinfo=timezone.utc)

    def make(source_id, doc_type, number, title, body, state=None, posts=()):
        return SourceDoc(
            repo="o/r", source_id=source_id, doc_type=doc_type, number=number, title=title,
            url=f"https://github.com/o/r/{source_id.replace(':', '/')}", author="alice", labels=[],
            state=state, created_at=now, updated_at=now, body=body,
            posts=[Post(author="bob", body=p, created_at=now) for p in posts],
        )

    return [
        make("doc:docs/config.md", "doc", None, "Configuration",
             "# Settings\nSet the request timeout in config.toml using the timeout key."),
        make("doc:docs/auth.md", "doc", None, "Authentication",
             "# JWT auth\nUse OAuth2 with JWT tokens for login."),
        make("issue:10", "issue", 10, "Background tasks broken",
             "Background tasks stopped running after upgrade.", state="open", posts=["Same here."]),
        make("issue:11", "issue", 11, "ERR_TIMEOUT on client",
             "ERR_TIMEOUT raised when the client waits too long.", state="closed"),
        make("pr:12", "pr", 12, "Fix background task scheduling",
             "Reschedules background tasks after upgrade.", state="merged"),
        make("issue:13", "issue", 13, "Login fails with JWT", "JWT login returns 401.", state="open"),
    ]


@pytest.fixture
def seeded_db(db):
    from copilot.ingest.pipeline import ingest_documents
    from fakes import FakeEmbedder

    ingest_documents(db, FakeEmbedder(), _seed_docs())
    return db
