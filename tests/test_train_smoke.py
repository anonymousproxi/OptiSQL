"""End-to-end check of train/sft.py on a tiny model. Opt in with SMOKE=1.

Downloads Qwen2.5-Coder-0.5B-Instruct (about 1 GB), trains 3 steps on CPU or GPU, and checks
the run record. It proves the pipeline works; the loss values mean nothing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from data.prompt import build_prompt
from tests.conftest import CONCERT_DDL, make_db
from train.config import load_config
from train.data import load_jsonl

pytestmark = pytest.mark.skipif(not os.environ.get("SMOKE"), reason="set SMOKE=1 to run")

CONFIG = Path(__file__).resolve().parent.parent / "train" / "configs" / "smoke.toml"

QUESTIONS = [
    ("How many singers are there?", "SELECT count(*) FROM singer"),
    ("List all singer names.", "SELECT name FROM singer"),
    ("What is the average age of singers?", "SELECT avg(age) FROM singer"),
    ("Which singers are from France?", "SELECT name FROM singer WHERE country = 'France'"),
    ("List concert titles.", "SELECT title FROM concert"),
    ("Who is the oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
    ("How many concerts are there?", "SELECT count(*) FROM concert"),
    ("List countries without repeats.", "SELECT DISTINCT country FROM singer"),
]


def write_rows(path: Path, db: Path, n: int, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as fh:
        for i in range(n):
            q, sql = QUESTIONS[i % len(QUESTIONS)]
            q = f"{q} (variant {i})"
            row = {"id": f"{prefix}{i}", "source": "fixture", "db_id": "concert", "question": q,
                   "evidence": "", "sql": sql, "messages": build_prompt(db, q, "", answer_sql=sql)}
            fh.write(json.dumps(row) + "\n")


def test_three_steps_end_to_end(tmp_path: Path) -> None:
    from train.sft import train

    db = tmp_path / "db" / "concert.sqlite"
    make_db(db, CONCERT_DDL, {"singer": [(1, "Ana", "France", 30), (2, "Bo", "Japan", 25)],
                               "concert": [(1, "Spring", 1)]})
    write_rows(tmp_path / "data" / "processed" / "sft_train.jsonl", db, 24, "t")
    write_rows(tmp_path / "data" / "processed" / "sft_val.jsonl", db, 8, "v")

    cfg = load_config(CONFIG)
    result = train(cfg, root=tmp_path)

    out = tmp_path / cfg.output_dir
    assert result["steps"] == 3
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "run_meta.json").exists() and (out / "data_stats.json").exists()

    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in log if "loss" in r]
    assert losses and all(0 < x < 100 for x in losses)
    assert "eval_loss" in result["eval_metrics"]

    stats = json.loads((out / "data_stats.json").read_text())
    assert stats["train"]["kept"] == 16 and stats["val"]["kept"] == 8
    assert stats["boundary_check"]["mismatches"] == 0
    assert 0 < stats["train"]["completion_tokens_total"] < stats["train"]["tokens_total"]

    # the examples really are the shared prompt format: the system prompt is present
    first = load_jsonl(tmp_path / "data/processed/sft_train.jsonl")[0]
    assert "SQLite analyst" in first["messages"][0]["content"]
