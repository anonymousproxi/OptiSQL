"""Tiny Spider and BIRD layouts with planted duplicates, broken SQL and eval leakage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


def make_db(path: Path, ddl: str, rows: dict[str, list[tuple[object, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(ddl)
    for table, values in rows.items():
        if values:
            marks = ",".join("?" * len(values[0]))
            conn.executemany(f'INSERT INTO "{table}" VALUES ({marks})', values)
    conn.commit()
    conn.close()


CONCERT_DDL = """
CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INT);
CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, title TEXT, singer_id INT,
                      FOREIGN KEY (singer_id) REFERENCES singer(singer_id));
"""
PETS_DDL = "CREATE TABLE pet (pet_id INTEGER PRIMARY KEY, kind TEXT, weight REAL);"
ODD_DDL = 'CREATE TABLE "order items" ("item id" INT, qty INT, PRIMARY KEY ("item id", qty));'


def _dump(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    sp = root / "spider"
    db = sp / "database"
    make_db(db / "concert" / "concert.sqlite", CONCERT_DDL,
            {"singer": [(1, "Ana", "France", 30), (2, "Bo", "Japan", 25)],
             "concert": [(1, "Spring", 1)]})
    make_db(db / "odd" / "odd.sqlite", ODD_DDL, {"order items": [(1, 2)]})
    make_db(db / "pets" / "pets.sqlite", PETS_DDL, {"pet": [(1, "dog", 12.5)]})
    # Same tables/columns as the dev database "pets", under another name.
    make_db(db / "pets_copy" / "pets_copy.sqlite", PETS_DDL, {"pet": [(1, "cat", 3.0)]})

    _dump(sp / "train_spider.json", [
        {"db_id": "concert", "question": "How many  singers are there?",
         "query": "SELECT count(*) FROM singer;"},
        {"db_id": "concert", "question": "How many singers are there ?",  # exact dup
         "query": "SELECT  count(*)  FROM singer"},
        {"db_id": "concert", "question": "How many singers are there?",  # conflicting label
         "query": "SELECT count(singer_id) FROM singer"},
        {"db_id": "concert", "question": "List singer names.", "query": "SELECT name FROM singer"},
        {"db_id": "concert", "question": "Broken one.", "query": "SELECT nope FROM singer"},
        {"db_id": "concert", "question": "What is the heaviest pet?",  # matches a dev question
         "query": "SELECT max(age) FROM singer"},
        {"db_id": "pets", "question": "Pets in train but dev db.", "query": "SELECT * FROM pet"},
        {"db_id": "pets_copy", "question": "Kinds of pet.", "query": "SELECT kind FROM pet"},
        {"db_id": "missing_db", "question": "No database file.", "query": "SELECT 1"},
    ])
    _dump(sp / "train_others.json", [
        {"db_id": "odd", "question": "Total quantity?",
         "query": 'SELECT sum(qty) FROM "order items"'},
    ])
    _dump(sp / "dev.json", [
        {"db_id": "pets", "question": "What is the heaviest pet?",
         "query": "SELECT max(weight) FROM pet"},
    ])

    bird = root / "bird"
    make_db(bird / "train" / "train_databases" / "shop" / "shop.sqlite",
            "CREATE TABLE item (id INTEGER PRIMARY KEY, label TEXT, stock INT);",
            {"item": [(1, "pen", 10), (2, "ink", 0)]})
    make_db(bird / "dev" / "dev_databases" / "school" / "school.sqlite",
            "CREATE TABLE student (id INT, name TEXT);", {"student": [(1, "Al")]})
    _dump(bird / "train" / "train.json", [
        {"db_id": "shop", "question": "Items out of stock?", "evidence": "out of stock: stock = 0",
         "SQL": "SELECT label FROM item WHERE stock = 0"},
        {"db_id": "shop", "question": "Items out of stock?", "evidence": "",
         "SQL": "SELECT label FROM item WHERE stock < 1"},
    ])
    _dump(bird / "dev" / "dev.json", [
        {"question_id": 0, "db_id": "school", "question": "Name of student 1?", "evidence": "",
         "SQL": "SELECT name FROM student WHERE id = 1", "difficulty": "simple"},
    ])
    return root
