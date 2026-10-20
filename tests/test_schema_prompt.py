from __future__ import annotations

from pathlib import Path

from data.prompt import SYSTEM_PROMPT, build_messages, extract_sql
from data.schema import quote_ident, schema_fingerprint, serialize_schema


def test_serialize_schema_includes_keys(raw: Path) -> None:
    text = serialize_schema(raw / "spider/database/concert/concert.sqlite")
    assert "CREATE TABLE singer (\n  singer_id INTEGER PRIMARY KEY," in text
    assert "FOREIGN KEY (singer_id) REFERENCES singer(singer_id)" in text
    assert "Example rows" not in text


def test_serialize_schema_quotes_and_composite_keys(raw: Path) -> None:
    text = serialize_schema(raw / "spider/database/odd/odd.sqlite", 2)
    assert 'CREATE TABLE "order items"' in text
    assert 'PRIMARY KEY ("item id", qty)' in text
    assert "-- Example rows (item id, qty):\n--   1, 2" in text


def test_sample_values_are_truncated(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "long.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (s TEXT, b BLOB, n INT)")
        conn.execute("INSERT INTO t VALUES (?, ?, NULL)", ("x" * 200 + "\nline", b"\x00"))
    text = serialize_schema(path, 1)
    assert "x" * 40 + "..." in text and "x" * 41 not in text
    assert "<blob>" in text and "NULL" in text


def test_fingerprint_ignores_name_and_data(raw: Path) -> None:
    db = raw / "spider/database"
    assert schema_fingerprint(db / "pets/pets.sqlite") == \
        schema_fingerprint(db / "pets_copy/pets_copy.sqlite")
    assert schema_fingerprint(db / "pets/pets.sqlite") != \
        schema_fingerprint(db / "concert/concert.sqlite")


def test_quote_ident() -> None:
    assert quote_ident("abc_1") == "abc_1"
    assert quote_ident('we"ird name') == '"we""ird name"'


def test_build_messages() -> None:
    msgs = build_messages("CREATE TABLE t (a INT);", " Q? ", evidence="hint here",
                          answer_sql="SELECT a FROM t")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"].endswith("Hint: hint here\n\nQuestion: Q?")
    assert "Hint" not in build_messages("s", "q")[1]["content"]
    assert len(build_messages("s", "q")) == 2


def test_extract_sql() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```\nthis returns one") == "SELECT 1"
    assert extract_sql("  SELECT a FROM t;  ") == "SELECT a FROM t"
    assert extract_sql("Here:\n```\nSELECT 2\n```") == "SELECT 2"
