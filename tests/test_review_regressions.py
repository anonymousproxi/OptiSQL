"""Regression tests for issues found in the M1 review."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data.prepare import clean_sql, default_splits, main, question_key, run
from data.prompt import SCHEMA_SAMPLE_ROWS, build_prompt, extract_sql
from data.schema import quote_ident, serialize_schema, sql_literal
from tests.conftest import make_db


def test_keyword_table_and_column_names(tmp_path: Path) -> None:
    db = tmp_path / "kw.sqlite"
    make_db(db, 'CREATE TABLE "order" (id INT, "group" TEXT, "from" TEXT, '
                'FOREIGN KEY (id) REFERENCES "order"(id));',
            {"order": [(1, "a", "b")]})
    text = serialize_schema(db, 1)
    assert 'CREATE TABLE "order" (' in text
    assert '"group" TEXT' in text and '"from" TEXT' in text
    assert 'REFERENCES "order"(id)' in text
    assert quote_ident("Select") == '"Select"'
    assert quote_ident("singer") == "singer"


def test_clean_sql_keeps_string_literals() -> None:
    assert clean_sql("SELECT  *\n FROM t WHERE name = 'New  York' ;") == \
        "SELECT * FROM t WHERE name = 'New  York'"
    assert clean_sql("SELECT \"a  b\" FROM [x  y] WHERE s = 'it''s  ok'") == \
        "SELECT \"a  b\" FROM [x  y] WHERE s = 'it''s  ok'"


def test_missing_eval_dbs_stop_the_run(raw: Path, tmp_path: Path) -> None:
    (raw / "bird/dev/dev_databases/school/school.sqlite").unlink()
    train, evals = default_splits(raw / "spider", raw / "bird")
    with pytest.raises(FileNotFoundError, match="eval database files are missing"):
        run(train, evals, tmp_path / "o")
    summary = run(train, evals, tmp_path / "o", allow_missing_eval_dbs=True)
    assert summary["eval_dbs_missing"] == ["bird:school"]
    assert "WEAKER" in summary["leakage_check"]
    assert main(["--spider-dir", str(raw / "spider"), "--bird-dir", str(raw / "bird"),
                 "--out", str(tmp_path / "o2")]) == 2


def _add_bird_dev_db(raw: Path, db_id: str, ddl: str) -> None:
    make_db(raw / f"bird/dev/dev_databases/{db_id}/{db_id}.sqlite", ddl, {})
    dev = json.loads((raw / "bird/dev/dev.json").read_text())
    dev.append({"question_id": len(dev), "db_id": db_id, "question": f"About {db_id}?",
                "evidence": "", "SQL": "SELECT 1", "difficulty": "simple"})
    (raw / "bird/dev/dev.json").write_text(json.dumps(dev))


def test_same_db_id_across_sources_is_a_leak(raw: Path, tmp_path: Path) -> None:
    # BIRD dev has a database also called "concert" (different schema) -> Spider train drops it.
    _add_bird_dev_db(raw, "concert", "CREATE TABLE gig (id INT, venue TEXT);")
    train, evals = default_splits(raw / "spider", raw / "bird")
    summary = run(train, evals, tmp_path / "o", val_fraction=0.0)
    assert summary["counts"]["spider.dropped.leak_eval_db"] >= 3


def test_near_copy_schema_is_a_leak(raw: Path, tmp_path: Path) -> None:
    # Eval db = the Spider train "concert" schema plus an extra table, under another name.
    _add_bird_dev_db(raw, "music_events", """
        CREATE TABLE singer (singer_id INT, name TEXT, country TEXT, age INT);
        CREATE TABLE concert (concert_id INT, title TEXT, singer_id INT);
        CREATE TABLE ticket (ticket_id INT, concert_id INT, price REAL, seat TEXT,
                             buyer TEXT, sold_at TEXT);
    """)
    train, evals = default_splits(raw / "spider", raw / "bird")
    out = tmp_path / "o"
    summary = run(train, evals, out, val_fraction=0.0)
    assert summary["counts"]["spider.dropped.leak_eval_schema_similar"] >= 2
    dropped = [json.loads(x) for x in (out / "dropped.jsonl").read_text().splitlines()]
    assert any("bird:music_events" in d["detail"] for d in dropped)
    kept = [json.loads(x) for x in (out / "sft_train.jsonl").read_text().splitlines()]
    assert all(r["db_id"] != "concert" for r in kept)


def test_question_key_normalizes_unicode_and_punctuation() -> None:
    assert question_key("What’s the  NAME?") == question_key("what's the name")
    assert question_key("ﬁrst singer") == question_key("first singer")


def test_build_prompt_fixes_sample_rows(raw: Path) -> None:
    db = raw / "spider/database/concert/concert.sqlite"
    content = build_prompt(db, "q")[1]["content"]
    assert content.startswith("Database schema:\n" + serialize_schema(db, SCHEMA_SAMPLE_ROWS))
    assert "-- Example rows" in content


def test_training_prompts_match_build_prompt(raw: Path, tmp_path: Path) -> None:
    train, evals = default_splits(raw / "spider", raw / "bird")
    out = tmp_path / "o"
    run(train, evals, out, val_fraction=0.0)
    row = json.loads((out / "sft_train.jsonl").read_text().splitlines()[0])
    db = raw / "spider/database" / row["db_id"] / f"{row['db_id']}.sqlite"
    if row["source"] == "bird":
        db = raw / "bird/train/train_databases" / row["db_id"] / f"{row['db_id']}.sqlite"
    assert row["messages"] == build_prompt(db, row["question"], row["evidence"], row["sql"])


@pytest.mark.parametrize("reply,expected", [
    ("```sqlite\nSELECT 1\n```", "SELECT 1"),
    ("```sql\nSELECT 1;\n```\nthis returns one", "SELECT 1"),
    ("```sql\nSELECT a FROM t WHERE", "SELECT a FROM t WHERE"),
    ("The answer is:\nSELECT 1;", "SELECT 1"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("SELECT a FROM t;", "SELECT a FROM t"),
])
def test_extract_sql_cases(reply: str, expected: str) -> None:
    assert extract_sql(reply) == expected


def test_sample_values_are_sql_literals() -> None:
    assert sql_literal("it's") == "'it''s'"
    assert sql_literal(None) == "NULL"
    assert sql_literal(2.5) == "2.5"


def test_composite_foreign_key(tmp_path: Path) -> None:
    db = tmp_path / "fk.sqlite"
    make_db(db, """
        CREATE TABLE a (p INT, q INT, PRIMARY KEY (p, q));
        CREATE TABLE b (x INT, y INT, FOREIGN KEY (x, y) REFERENCES a(p, q));
    """, {})
    text = serialize_schema(db, 0)
    assert "FOREIGN KEY (x, y) REFERENCES a(p, q)" in text
    assert text.count("FOREIGN KEY") == 1


def test_fts_shadow_tables_skipped_and_views_included(tmp_path: Path) -> None:
    db = tmp_path / "v.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE doc (id INT, body TEXT);
        CREATE VIRTUAL TABLE f USING fts5(body);
        CREATE VIEW long_docs AS SELECT id FROM doc WHERE length(body) > 10;
    """)
    conn.close()
    text = serialize_schema(db, 0)
    assert "CREATE TABLE f (" in text
    assert "f_data" not in text and "f_idx" not in text and "f_content" not in text
    assert "CREATE VIEW long_docs AS SELECT id FROM doc WHERE length(body) > 10;" in text


def test_regular_table_with_virtual_prefix_is_kept(tmp_path: Path) -> None:
    db = tmp_path / "p.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE VIRTUAL TABLE f USING fts5(body);
        CREATE TABLE f_notes (id INT, note TEXT);
    """)
    conn.close()
    text = serialize_schema(db, 0)
    assert "CREATE TABLE f_notes (" in text
    assert "f_data" not in text and "f_config" not in text


def test_broken_view_does_not_stop_the_run(tmp_path: Path) -> None:
    db = tmp_path / "bv.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE t (a INT);
        CREATE VIEW v AS SELECT a FROM t;
        DROP TABLE t;
        CREATE TABLE u (b INT);
    """)
    conn.close()
    text = serialize_schema(db, 1)
    assert "CREATE TABLE u (" in text
    assert "-- View is broken" in text
