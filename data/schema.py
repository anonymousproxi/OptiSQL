"""Serialize a SQLite database schema into compact CREATE TABLE text for prompts.

The DDL is rebuilt from PRAGMA metadata rather than copied from sqlite_master, because the
original DDL in Spider and BIRD varies a lot in style (comments, quoting, odd whitespace).
Rebuilding it gives every prompt the same format.
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
_MAX_VALUE_CHARS = 40


def quote_ident(name: str) -> str:
    if _SIMPLE_IDENT.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str | None


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]
    sample_rows: tuple[tuple[Any, ...], ...] = ()


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def read_tables(db_path: str | Path, sample_rows: int = 0) -> list[Table]:
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    conn = _connect_ro(db_path)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )]
        tables: list[Table] = []
        for name in names:
            q = quote_ident(name)
            cols = tuple(
                Column(name=r[1], type=(r[2] or "").upper(), primary_key=bool(r[5]))
                for r in conn.execute(f"PRAGMA table_info({q})")
            )
            fks = tuple(
                ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4])
                for r in conn.execute(f"PRAGMA foreign_key_list({q})")
            )
            rows: tuple[tuple[Any, ...], ...] = ()
            if sample_rows > 0:
                rows = tuple(tuple(r) for r in conn.execute(
                    f"SELECT * FROM {q} LIMIT ?", (sample_rows,)
                ))
            tables.append(Table(name=name, columns=cols, foreign_keys=fks, sample_rows=rows))
        return tables
    finally:
        conn.close()


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "<blob>"
    text = str(value).replace("\n", " ")
    if len(text) > _MAX_VALUE_CHARS:
        text = text[:_MAX_VALUE_CHARS] + "..."
    return repr(text) if isinstance(value, str) else text


def table_ddl(table: Table) -> str:
    parts: list[str] = []
    pk_cols = [c for c in table.columns if c.primary_key]
    for col in table.columns:
        line = quote_ident(col.name)
        if col.type:
            line += f" {col.type}"
        if col.primary_key and len(pk_cols) == 1:
            line += " PRIMARY KEY"
        parts.append(line)
    if len(pk_cols) > 1:
        parts.append("PRIMARY KEY (" + ", ".join(quote_ident(c.name) for c in pk_cols) + ")")
    for fk in table.foreign_keys:
        ref = quote_ident(fk.ref_table)
        if fk.ref_column:
            ref += f"({quote_ident(fk.ref_column)})"
        parts.append(f"FOREIGN KEY ({quote_ident(fk.column)}) REFERENCES {ref}")
    ddl = f"CREATE TABLE {quote_ident(table.name)} (\n  " + ",\n  ".join(parts) + "\n);"
    if table.sample_rows:
        header = ", ".join(c.name for c in table.columns)
        rows = "\n".join(
            "--   " + ", ".join(_format_value(v) for v in row) for row in table.sample_rows
        )
        ddl += f"\n-- Example rows ({header}):\n{rows}"
    return ddl


@lru_cache(maxsize=512)
def serialize_schema(db_path: str | Path, sample_rows: int = 0) -> str:
    """CREATE TABLE text for every table, optionally followed by a few example rows."""
    return "\n\n".join(table_ddl(t) for t in read_tables(db_path, sample_rows))


@lru_cache(maxsize=2048)
def schema_fingerprint(db_path: str | Path) -> str:
    """Hash of table and column names (case-insensitive). Two databases with the same
    fingerprint are treated as the same database for leakage checks, whatever their db_id."""
    items = sorted(
        (t.name.lower(), tuple(sorted(c.name.lower() for c in t.columns)))
        for t in read_tables(db_path)
    )
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]
