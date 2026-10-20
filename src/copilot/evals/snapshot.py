"""Eval data snapshot: the index + LLM cache, so CI can evaluate without re-syncing GitHub or
re-embedding (and, with the cache, re-run unchanged cases for free).

Format: a .tar.gz with manifest.json and one `COPY ... TO STDOUT` file per table. COPY is
plain text that any Postgres version reads, unlike pg_dump archives (tied to the server version).
"""

import io
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psycopg

# Dependency order: chunks reference documents. Threads/actions are runtime state, never snapshotted.
SNAPSHOT_TABLES = ("documents", "chunks", "sync_state", "llm_cache", "llm_usage")
FORMAT = 1
_BLOCK = 1 << 20


class SnapshotError(Exception):
    pass


def _columns(conn: psycopg.Connection, table: str) -> list[str]:
    rows = conn.execute("select column_name from information_schema.columns where table_schema = 'public' "
                        "and table_name = %s order by ordinal_position", (table,)).fetchall()
    return [r["column_name"] for r in rows]


def _copy_sql(table: str, columns: list[str], direction: str) -> str:
    cols = ", ".join(f'"{c}"' for c in columns)
    return f'COPY {table} ({cols}) {direction}'


def export_snapshot(conn: psycopg.Connection, path: Path) -> dict:
    """Write the snapshot to `path` and return its manifest."""
    manifest = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(), "tables": {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(path, "w:gz") as tar:
        for table in SNAPSHOT_TABLES:
            columns = _columns(conn, table)
            data = Path(tmp) / f"{table}.copy"
            with data.open("wb") as f, conn.cursor().copy(_copy_sql(table, columns, "TO STDOUT")) as copy:
                for block in copy:
                    f.write(block)
            rows = conn.execute(f"select count(*) as n from {table}").fetchone()["n"]
            manifest["tables"][table] = {"columns": columns, "rows": rows}
            tar.add(data, arcname=f"{table}.copy")
        conn.commit()  # end the read transaction
        body = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return manifest


def import_snapshot(conn: psycopg.Connection, path: Path) -> dict:
    """Replace the snapshot tables with the snapshot's rows, all in one transaction: on any problem
    nothing changes. Only known member names are read (nothing is extracted to disk)."""
    with tarfile.open(path, "r:gz") as tar:
        manifest = _read_manifest(tar)
        for table in SNAPSHOT_TABLES:
            missing = set(manifest["tables"][table]["columns"]) - set(_columns(conn, table))
            if missing:
                raise SnapshotError(f"{table}: database lacks columns {sorted(missing)}; run `copilot init-db`")
        try:
            conn.execute(f"TRUNCATE {', '.join(SNAPSHOT_TABLES)} RESTART IDENTITY")
            for table in SNAPSHOT_TABLES:
                spec = manifest["tables"][table]
                member = tar.extractfile(f"{table}.copy")
                with conn.cursor().copy(_copy_sql(table, spec["columns"], "FROM STDIN")) as copy:
                    while block := member.read(_BLOCK):
                        copy.write(block)
                rows = conn.execute(f"select count(*) as n from {table}").fetchone()["n"]
                if rows != spec["rows"]:
                    raise SnapshotError(f"{table}: snapshot says {spec['rows']} rows, got {rows} (corrupt file?)")
                if "id" in spec["columns"]:
                    # Restored ids are explicit: move the sequence past them so new inserts don't collide.
                    conn.execute(f"select setval(pg_get_serial_sequence('{table}', 'id'), "
                                 f"coalesce(max(id), 1), max(id) is not null) from {table}")
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
    return manifest


def _read_manifest(tar: tarfile.TarFile) -> dict:
    try:
        manifest = json.load(tar.extractfile("manifest.json"))
    except KeyError as e:
        raise SnapshotError("not an eval snapshot: manifest.json missing") from e
    if manifest.get("format") != FORMAT:
        raise SnapshotError(f"unsupported snapshot format {manifest.get('format')!r} (expected {FORMAT})")
    if set(manifest.get("tables", {})) != set(SNAPSHOT_TABLES):
        raise SnapshotError(f"snapshot tables {sorted(manifest.get('tables', {}))} != {list(SNAPSHOT_TABLES)}")
    return manifest
