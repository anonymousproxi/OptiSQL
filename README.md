# sql-finetune

Work in progress. See CLAUDE.md for the plan and PROGRESS.md for status.

- `data/`: M1 data preparation (schema serialization, shared prompt format, cleaning, dedup,
  leakage checks). See [docs/DATA.md](docs/DATA.md).
- Evaluation uses `t2s_eval` from `../text2sql/eval` (install with `pip install -r requirements-data.txt`).

Tests: `pytest` (offline; builds tiny fixture databases).
