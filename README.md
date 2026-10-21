# sql-finetune

Work in progress. See CLAUDE.md for the plan and PROGRESS.md for status.

- `data/`: M1 data preparation (schema serialization, shared prompt format, cleaning, dedup,
  leakage checks). See [docs/DATA.md](docs/DATA.md).
- Evaluation uses `t2s_eval` from `../EvalSQL/eval` (install with `pip install -r requirements-data.txt`).

- `train/`: M2 supervised fine-tuning scripts (QLoRA), configs in `train/configs/`, Colab notebook in
  `notebooks/`. See [docs/TRAINING.md](docs/TRAINING.md). No training results yet.

Tests: `pytest` (offline; builds tiny fixture databases). `SMOKE=1 pytest tests/test_train_smoke.py`
runs a real 3-step training on a 0.5B model (downloads about 1 GB).
