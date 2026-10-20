"""Build SFT data from the Spider and BIRD training splits.

Steps: load -> clean -> drop gold SQL that fails to execute -> remove duplicates -> remove any
overlap with the evaluation splits -> group train/val split by database -> write JSONL.

Usage (from sql-finetune/):
    python -m data.prepare --spider-dir data/raw/spider --bird-dir data/raw/bird \
        --out data/processed

Every dropped example is counted in stats.json and listed in dropped.jsonl, so nothing
disappears silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from t2s_eval import Example, execute_sql, load_examples

from data.prompt import PROMPT_VERSION, build_messages
from data.schema import schema_fingerprint, serialize_schema


@dataclass(frozen=True)
class SplitFile:
    """One gold JSON file plus the directory holding its databases."""

    source: str  # "spider" or "bird"
    name: str  # e.g. "train_spider", "dev"
    gold: Path
    db_root: Path


@dataclass
class Record:
    id: str
    source: str
    db_id: str
    question: str
    evidence: str
    sql: str
    db_path: Path
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "source": self.source, "db_id": self.db_id,
                "question": self.question, "evidence": self.evidence, "sql": self.sql,
                "messages": self.messages}


_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s?.!]+$")


def clean_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def clean_sql(sql: str) -> str:
    return _WS.sub(" ", sql).strip().rstrip(";").strip()


def question_key(question: str) -> str:
    """Normalized question used for duplicate and leakage matching."""
    return _TRAILING_PUNCT.sub("", clean_text(question).lower())


def default_splits(spider_dir: Path | None, bird_dir: Path | None
                   ) -> tuple[list[SplitFile], list[SplitFile]]:
    """Train and eval files in the official layouts. Optional files are skipped if absent."""
    train: list[SplitFile] = []
    evals: list[SplitFile] = []
    if spider_dir is not None:
        db = spider_dir / "database"
        train += [SplitFile("spider", "train_spider", spider_dir / "train_spider.json", db),
                  SplitFile("spider", "train_others", spider_dir / "train_others.json", db)]
        evals.append(SplitFile("spider", "dev", spider_dir / "dev.json", db))
        if (spider_dir / "test.json").is_file():
            test_db = spider_dir / "test_database"
            evals.append(SplitFile("spider", "test", spider_dir / "test.json",
                                   test_db if test_db.is_dir() else db))
    if bird_dir is not None:
        train.append(SplitFile("bird", "train", bird_dir / "train" / "train.json",
                               bird_dir / "train" / "train_databases"))
        evals.append(SplitFile("bird", "dev", bird_dir / "dev" / "dev.json",
                               bird_dir / "dev" / "dev_databases"))
    return train, evals


def _load(split: SplitFile) -> list[Example]:
    if not split.gold.is_file():
        raise FileNotFoundError(f"{split.source} {split.name}: missing {split.gold}")
    return load_examples(split.gold, split.db_root)


class Stats:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.dropped: list[dict[str, str]] = []

    def drop(self, rec: Record, reason: str, detail: str = "") -> None:
        self.counts[f"{rec.source}.dropped.{reason}"] += 1
        self.dropped.append({"id": rec.id, "source": rec.source, "db_id": rec.db_id,
                             "reason": reason, "detail": detail[:300],
                             "question": rec.question})


def load_train(splits: Sequence[SplitFile], stats: Stats) -> list[Record]:
    records: list[Record] = []
    for split in splits:
        examples = _load(split)
        stats.counts[f"{split.source}.loaded.{split.name}"] += len(examples)
        for ex in examples:
            rec = Record(id=f"{split.source}-{split.name}-{ex.id}", source=split.source,
                         db_id=ex.db_id, question=clean_text(ex.question),
                         evidence=clean_text(ex.evidence), sql=clean_sql(ex.gold_sql),
                         db_path=ex.db_path)
            if not rec.question or not rec.sql:
                stats.drop(rec, "empty")
                continue
            records.append(rec)
    return records


def drop_failing_sql(records: Iterable[Record], stats: Stats, timeout_s: float) -> list[Record]:
    kept: list[Record] = []
    for rec in records:
        if not rec.db_path.is_file():
            stats.drop(rec, "db_missing", str(rec.db_path))
            continue
        res = execute_sql(rec.db_path, rec.sql, timeout_s=timeout_s)
        if res.ok:
            kept.append(rec)
        else:
            stats.drop(rec, f"gold_{res.error_kind}", res.error or "")
    return kept


def dedupe(records: Iterable[Record], stats: Stats) -> list[Record]:
    """Keep the first of any exact (db, question, sql) duplicate, and the first of any
    (db, question) pair that appears with different SQL (an ambiguous label)."""
    seen_exact: set[tuple[str, str, str, str]] = set()
    seen_question: dict[tuple[str, str, str], str] = {}
    kept: list[Record] = []
    for rec in records:
        q = question_key(rec.question + " " + rec.evidence)
        exact = (rec.source, rec.db_id, q, rec.sql.lower())
        if exact in seen_exact:
            stats.drop(rec, "duplicate")
            continue
        qkey = (rec.source, rec.db_id, q)
        if qkey in seen_question:
            stats.drop(rec, "conflicting_duplicate", f"kept {seen_question[qkey]}")
            continue
        seen_exact.add(exact)
        seen_question[qkey] = rec.id
        kept.append(rec)
    return kept


@dataclass
class EvalIndex:
    questions: set[str]
    db_ids: set[tuple[str, str]]
    fingerprints: dict[str, str]  # fingerprint -> "source:db_id" of the eval database
    counts: dict[str, int]


def build_eval_index(splits: Sequence[SplitFile]) -> EvalIndex:
    questions: set[str] = set()
    db_ids: set[tuple[str, str]] = set()
    fingerprints: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in splits:
        examples = _load(split)
        counts[f"{split.source}.{split.name}"] = len(examples)
        for ex in examples:
            questions.add(question_key(ex.question))
            if (split.source, ex.db_id) not in db_ids and ex.db_path.is_file():
                fingerprints[schema_fingerprint(ex.db_path)] = f"{split.source}:{ex.db_id}"
            db_ids.add((split.source, ex.db_id))
    return EvalIndex(questions, db_ids, fingerprints, counts)


def drop_eval_overlap(records: Iterable[Record], index: EvalIndex, stats: Stats) -> list[Record]:
    """Drop a training example if its database is an eval database (same source and db_id, or
    the same table/column fingerprint under any name) or its question matches any eval
    question."""
    kept: list[Record] = []
    for rec in records:
        if (rec.source, rec.db_id) in index.db_ids:
            stats.drop(rec, "leak_eval_db", rec.db_id)
            continue
        fp = schema_fingerprint(rec.db_path)
        if fp in index.fingerprints:
            stats.drop(rec, "leak_eval_schema", index.fingerprints[fp])
            continue
        if question_key(rec.question) in index.questions:
            stats.drop(rec, "leak_eval_question")
            continue
        kept.append(rec)
    return kept


def attach_prompts(records: Iterable[Record], stats: Stats, sample_rows: int,
                   max_prompt_chars: int) -> list[Record]:
    kept: list[Record] = []
    for rec in records:
        schema = serialize_schema(rec.db_path, sample_rows)
        rec.messages = build_messages(schema, rec.question, rec.evidence, answer_sql=rec.sql)
        prompt_chars = sum(len(m["content"]) for m in rec.messages)
        if prompt_chars > max_prompt_chars:
            stats.drop(rec, "too_long", str(prompt_chars))
            continue
        kept.append(rec)
    return kept


def split_by_db(records: Sequence[Record], val_fraction: float, seed: int
                ) -> tuple[list[Record], list[Record]]:
    """Hold out whole databases for validation so val loss measures unseen schemas."""
    dbs = sorted({(r.source, r.db_id) for r in records})
    n_val = round(len(dbs) * val_fraction)
    if val_fraction > 0 and n_val == 0 and len(dbs) > 1:
        n_val = 1
    val_dbs = set(random.Random(seed).sample(dbs, n_val)) if n_val else set()
    train = [r for r in records if (r.source, r.db_id) not in val_dbs]
    val = [r for r in records if (r.source, r.db_id) in val_dbs]
    return train, val


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _length_stats(records: Sequence[Record]) -> dict[str, float]:
    lengths = sorted(sum(len(m["content"]) for m in r.messages) for r in records)
    if not lengths:
        return {}
    return {"min": lengths[0], "median": statistics.median(lengths),
            "p95": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
            "max": lengths[-1]}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(train_splits: Sequence[SplitFile], eval_splits: Sequence[SplitFile], out: Path,
        sample_rows: int = 3, max_prompt_chars: int = 16_000, val_fraction: float = 0.05,
        seed: int = 0, skip_exec: bool = False, exec_timeout: float = 30.0) -> dict[str, Any]:
    stats = Stats()
    index = build_eval_index(eval_splits)
    records = load_train(train_splits, stats)
    if not skip_exec:
        records = drop_failing_sql(records, stats, exec_timeout)
    else:
        present = []
        for rec in records:
            if rec.db_path.is_file():
                present.append(rec)
            else:
                stats.drop(rec, "db_missing", str(rec.db_path))
        records = present
    records = dedupe(records, stats)
    records = drop_eval_overlap(records, index, stats)
    records = attach_prompts(records, stats, sample_rows, max_prompt_chars)
    train, val = split_by_db(records, val_fraction, seed)

    # Final guard: fail loudly if any overlap survived the filters above.
    leaked = [r.id for r in records if question_key(r.question) in index.questions
              or (r.source, r.db_id) in index.db_ids]
    if leaked:
        raise RuntimeError(f"leakage check failed for {len(leaked)} records: {leaked[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "sft_train.jsonl", (r.to_json() for r in train))
    _write_jsonl(out / "sft_val.jsonl", (r.to_json() for r in val))
    _write_jsonl(out / "dropped.jsonl", stats.dropped)

    per_source = {
        src: {"train": sum(r.source == src for r in train),
              "val": sum(r.source == src for r in val),
              "databases": len({r.db_id for r in records if r.source == src})}
        for src in sorted({s.source for s in train_splits})
    }
    summary: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "settings": {"sample_rows": sample_rows, "max_prompt_chars": max_prompt_chars,
                     "val_fraction": val_fraction, "seed": seed, "skip_exec": skip_exec,
                     "exec_timeout": exec_timeout},
        "inputs": {f"{s.source}.{s.name}": {"path": str(s.gold), "sha256": _sha256(s.gold)}
                   for s in [*train_splits, *eval_splits]},
        "eval_examples_checked": index.counts,
        "counts": dict(sorted(stats.counts.items())),
        "output": {"train": len(train), "val": len(val), "per_source": per_source},
        "prompt_chars": _length_stats(records),
        "leakage_check": "passed: no train/val question or database overlaps the eval splits",
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spider-dir", type=Path, default=None,
                   help="unpacked Spider folder (train_spider.json, dev.json, database/)")
    p.add_argument("--bird-dir", type=Path, default=None,
                   help="folder with BIRD train/ and dev/ subfolders")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--sample-rows", type=int, default=3,
                   help="example rows per table in the schema text (0 to disable)")
    p.add_argument("--max-prompt-chars", type=int, default=16_000,
                   help="drop examples whose full chat text is longer than this")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-exec", action="store_true",
                   help="do not execute gold SQL (faster, but keeps broken labels)")
    p.add_argument("--exec-timeout", type=float, default=30.0)
    args = p.parse_args(argv)
    if args.spider_dir is None and args.bird_dir is None:
        p.error("give --spider-dir and/or --bird-dir")
    train, evals = default_splits(args.spider_dir, args.bird_dir)
    try:
        summary = run(train, evals, args.out, sample_rows=args.sample_rows,
                      max_prompt_chars=args.max_prompt_chars, val_fraction=args.val_fraction,
                      seed=args.seed, skip_exec=args.skip_exec, exec_timeout=args.exec_timeout)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"counts": summary["counts"], "output": summary["output"]}, indent=2))
    print(f"wrote {args.out}/sft_train.jsonl, sft_val.jsonl, dropped.jsonl, stats.json",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
