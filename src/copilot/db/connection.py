from importlib.resources import files

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


def connect(url: str) -> psycopg.Connection:
    conn = psycopg.connect(url, row_factory=dict_row)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # Fresh database: the vector type doesn't exist until apply_schema() runs.
        conn.rollback()
    return conn


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute(files("copilot.db").joinpath("schema.sql").read_text())
    conn.commit()
    register_vector(conn)
