from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data.prepare import default_splits, main, question_key, run, split_by_db


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.fixture
def result(raw: Path, tmp_path: Path) -> tuple[dict[str, Any], Path]:
    train, evals = default_splits(raw / "spider", raw / "bird")
    out = tmp_path / "out"
    return run(train, evals, out, val_fraction=0.0), out


def test_counts_and_drop_reasons(result: tuple[dict[str, Any], Path]) -> None:
    summary, out = result
    c = summary["counts"]
    assert c["spider.loaded.train_spider"] == 9
    assert c["spider.loaded.train_others"] == 1
    assert c["bird.loaded.train"] == 2
    assert c["spider.dropped.duplicate"] == 1
    assert c["spider.dropped.conflicting_duplicate"] == 1
    assert c["spider.dropped.gold_sql_error"] == 1
    assert c["spider.dropped.db_missing"] == 1
    assert c["spider.dropped.leak_eval_db"] == 1
    assert c["spider.dropped.leak_eval_schema"] == 1
    assert c["spider.dropped.leak_eval_question"] == 1
    dropped = _read(out / "dropped.jsonl")
    assert len(dropped) == sum(v for k, v in c.items() if ".dropped." in k)


def test_kept_examples_and_format(result: tuple[dict[str, Any], Path]) -> None:
    summary, out = result
    train = _read(out / "sft_train.jsonl")
    # spider: count(*), names, odd sum; bird: both rows differ by evidence so both are kept
    assert sorted(r["id"] for r in train) == [
        "bird-train-0", "bird-train-1",
        "spider-train_others-0", "spider-train_spider-0", "spider-train_spider-3",
    ]
    first = next(r for r in train if r["id"] == "spider-train_spider-0")
    assert first["sql"] == "SELECT count(*) FROM singer"
    assert first["question"] == "How many singers are there?"
    roles = [m["role"] for m in first["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert first["messages"][2]["content"] == first["sql"]
    assert "CREATE TABLE singer" in first["messages"][1]["content"]
    bird = next(r for r in train if r["id"] == "bird-train-0")
    assert "Hint: out of stock: stock = 0" in bird["messages"][1]["content"]
    assert summary["output"]["per_source"]["spider"]["train"] == 3


def test_no_overlap_with_eval(result: tuple[dict[str, Any], Path], raw: Path) -> None:
    _, out = result
    dev_q = {question_key(r["question"]) for r in json.loads((raw / "spider/dev.json").read_text())}
    for r in _read(out / "sft_train.jsonl"):
        assert question_key(r["question"]) not in dev_q
        assert r["db_id"] not in {"pets", "pets_copy", "school"}


def test_stats_file_records_inputs(result: tuple[dict[str, Any], Path]) -> None:
    _, out = result
    stats = json.loads((out / "stats.json").read_text())
    assert stats["prompt_version"] == "v1"
    assert len(stats["inputs"]["spider.train_spider"]["sha256"]) == 64
    assert stats["eval_examples_checked"] == {"spider.dev": 1, "bird.dev": 1}
    assert stats["prompt_chars"]["max"] >= stats["prompt_chars"]["min"] > 0


def test_too_long_prompts_are_dropped(raw: Path, tmp_path: Path) -> None:
    train, evals = default_splits(raw / "spider", raw / "bird")
    summary = run(train, evals, tmp_path / "o", max_prompt_chars=10, val_fraction=0.0)
    assert summary["output"]["train"] == 0
    assert summary["counts"]["spider.dropped.too_long"] == 3


def test_val_split_holds_out_whole_databases(raw: Path, tmp_path: Path) -> None:
    train, evals = default_splits(raw / "spider", raw / "bird")
    out = tmp_path / "o"
    run(train, evals, out, val_fraction=0.5, seed=1)
    tr, va = _read(out / "sft_train.jsonl"), _read(out / "sft_val.jsonl")
    assert tr and va
    assert not {(r["source"], r["db_id"]) for r in tr} & {(r["source"], r["db_id"]) for r in va}


def test_split_by_db_is_deterministic() -> None:
    from data.prepare import Record

    recs = [Record(id=str(i), source="s", db_id=f"db{i % 5}", question="q", evidence="",
                   sql="x", db_path=Path("x")) for i in range(20)]
    a, b = split_by_db(recs, 0.4, 3), split_by_db(recs, 0.4, 3)
    assert [r.id for r in a[1]] == [r.id for r in b[1]]
    assert len({r.db_id for r in a[1]}) == 2


def test_missing_eval_file_is_an_error(raw: Path, tmp_path: Path) -> None:
    (raw / "bird/dev/dev.json").unlink()
    code = main(["--spider-dir", str(raw / "spider"), "--bird-dir", str(raw / "bird"),
                 "--out", str(tmp_path / "o")])
    assert code == 2


def test_cli_spider_only(raw: Path, tmp_path: Path) -> None:
    code = main(["--spider-dir", str(raw / "spider"), "--out", str(tmp_path / "o"),
                 "--val-fraction", "0"])
    assert code == 0
    assert len(_read(tmp_path / "o" / "sft_train.jsonl")) == 3
