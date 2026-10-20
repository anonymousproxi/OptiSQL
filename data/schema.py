"""Serialize a SQLite database schema into compact CREATE TABLE text for prompts.

The DDL is rebuilt from PRAGMA metadata rather than copied from sqlite_master, because the
original DDL in Spider and BIRD varies a lot in style (comments, quoting, odd whitespace).
Rebuilding it gives every prompt the same format. Views are included with their original
definition, and shadow tables of virtual (FTS) tables are left out.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_VALUE_CHARS = 40

# SQLite's keyword list (https://sqlite.org/lang_keywords.html). Names that collide with one
# are quoted in the DDL so the model learns SQL that runs.
SQLITE_KEYWORDS = frozenset("""
ABORT ACTION ADD AFTER ALL ALTER ALWAYS ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN
BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS
CURRENT CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED
DELETE DESC DETACH DISTINCT DO DROP EACH ELSE END ESCAPE EXCEPT EXCLUDE EXCLUSIVE EXISTS
EXPLAIN FAIL FILTER FIRST FOLLOWING FOR FOREIGN FROM FULL GENERATED GLOB GROUP GROUPS HAVING
IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL
JOIN KEY LAST LEFT LIKE LIMIT MATCH MATERIALIZED NATURAL NO NOT NOTHING NOTNULL NULL NULLS OF
OFFSET ON OR ORDER OTHERS OUTER OVER PARTITION PLAN PRAGMA PRECEDING PRIMARY QUERY RAISE RANGE
RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE RESTRICT RETURNING RIGHT ROLLBACK
ROW ROWS SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TIES TO TRANSACTION TRIGGER UNBOUNDED
UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WINDOW WITH WITHOUT
""".split())


def always_quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_ident(name: str) -> str:
    """Quote only when needed: unusual characters or an SQLite keyword."""
    if _SIMPLE_IDENT.match(name) and name.upper() not in SQLITE_KEYWORDS:
        return name
    return always_quote(name)


def sql_literal(value: Any) -> str:
    """Render a sample value the way it would be written in SQLite."""
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "<blob>"
    if isinstance(value, int | float):
        return repr(value)
    text = str(value).replace("\r", " ").replace("\n", " ")
    if len(text) > MAX_VALUE_CHARS:
        text = text[:MAX_VALUE_CHARS] + "..."
    return "'" + text.replace("'", "''") + "'"


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    pk_position: int  # 0 if not part of the primary key, else 1-based position


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str | None, ...]


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]
    sample_rows: tuple[tuple[Any, ...], ...] = ()
    view_sql: str | None = None  # set for views


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def _foreign_keys(conn: sqlite3.Connection, table: str) -> tuple[ForeignKey, ...]:
    # PRAGMA foreign_key_list rows: (id, seq, table, from, to, ...). Rows sharing an id belong
    # to one (possibly composite) key.
    groups: dict[int, list[tuple[Any, ...]]] = {}
    for r in conn.execute(f"PRAGMA foreign_key_list({always_quote(table)})"):
        groups.setdefault(r[0], []).append(r)
    keys = []
    for _, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r[1])
        keys.append(ForeignKey(columns=tuple(r[3] for r in rows), ref_table=rows[0][2],
                               ref_columns=tuple(r[4] for r in rows)))
    return tuple(keys)


def read_tables(db_path: str | Path, sample_rows: int = 0) -> list[Table]:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    conn = _connect_ro(db_path)
    try:
        entries = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        virtual = [name for kind, name, sql in entries
                   if kind == "table" and (sql or "").upper().startswith("CREATE VIRTUAL")]
        tables: list[Table] = []
        for kind, name, sql in entries:
            if any(name != v and name.startswith(v + "_") for v in virtual):
                continue  # FTS/R-tree shadow table
            q = always_quote(name)
            cols = tuple(
                Column(name=r[1], type=(r[2] or "").upper(), pk_position=int(r[5]))
                for r in conn.execute(f"PRAGMA table_info({q})")
            )
            fks = _foreign_keys(conn, name) if kind == "table" else ()
            rows: tuple[tuple[Any, ...], ...] = ()
            if sample_rows > 0 and kind == "table":
                rows = tuple(tuple(r) for r in conn.execute(
                    f"SELECT * FROM {q} LIMIT ?", (sample_rows,)
                ))
            tables.append(Table(name=name, columns=cols, foreign_keys=fks, sample_rows=rows,
                                view_sql=" ".join((sql or "").split()) if kind == "view"
                                else None))
        return tables
    finally:
        conn.close()


def table_ddl(table: Table) -> str:
    if table.view_sql is not None:
        cols = ", ".join(quote_ident(c.name) for c in table.columns)
        return f"{table.view_sql};\n-- View columns: {cols}"
    parts: list[str] = []
    pk_cols = sorted((c for c in table.columns if c.pk_position), key=lambda c: c.pk_position)
    for col in table.columns:
        line = quote_ident(col.name)
        if col.type:
            line += f" {col.type}"
        if col.pk_position and len(pk_cols) == 1:
            line += " PRIMARY KEY"
        parts.append(line)
    if len(pk_cols) > 1:
        parts.append("PRIMARY KEY (" + ", ".join(quote_ident(c.name) for c in pk_cols) + ")")
    for fk in table.foreign_keys:
        ref = quote_ident(fk.ref_table)
        if all(fk.ref_columns):
            ref += "(" + ", ".join(quote_ident(c) for c in fk.ref_columns if c) + ")"
        cols = ", ".join(quote_ident(c) for c in fk.columns)
        parts.append(f"FOREIGN KEY ({cols}) REFERENCES {ref}")
    ddl = f"CREATE TABLE {quote_ident(table.name)} (\n  " + ",\n  ".join(parts) + "\n);"
    if table.sample_rows:
        header = ", ".join(quote_ident(c.name) for c in table.columns)
        rows = "\n".join(
            "--   " + ", ".join(sql_literal(v) for v in row) for row in table.sample_rows
        )
        ddl += f"\n-- Example rows ({header}):\n{rows}"
    return ddl


@lru_cache(maxsize=512)
def serialize_schema(db_path: str | Path, sample_rows: int) -> str:
    """CREATE TABLE text for every table and view, plus example rows when sample_rows > 0.

    Prompts should go through data.prompt.build_prompt, which fixes sample_rows for the
    current prompt version.
    """
    return "\n\n".join(table_ddl(t) for t in read_tables(db_path, sample_rows))


@lru_cache(maxsize=4096)
def schema_items(db_path: str | Path) -> frozenset[str]:
    """Lower-cased "table.column" names; the basis for exact and near-duplicate matching."""
    return frozenset(f"{t.name.lower()}.{c.name.lower()}"
                     for t in read_tables(db_path) for c in t.columns)


def schema_fingerprint(db_path: str | Path) -> str:
    """Hash of table and column names. Equal fingerprints mean the same schema."""
    return hashlib.sha256(repr(sorted(schema_items(db_path))).encode()).hexdigest()[:16]


def schema_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two schema_items sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def schema_containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Share of the smaller schema found in the larger one; catches a copy with extra tables."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
