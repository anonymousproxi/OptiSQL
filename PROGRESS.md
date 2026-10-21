# Progress: sql-finetune

Last updated: 2026-10-20. Status: M1 code-complete and reviewed. Waiting for the real Spider and BIRD downloads to produce the data.

## Milestones
| ID | Milestone | Status | Notes |
|---|---|---|---|
| M1 | Data preparation | Code complete, reviewed | Not yet run on real data (download blocked here); sizes not yet measured |
| M2 | Training scripts (SFT) | Code complete, reviewed by tests | Plain backend verified end to end on CPU (3 real steps, eval, checkpoint, adapter). Unsloth backend and 4-bit loading **not yet run** (need a CUDA GPU). See docs/TRAINING.md |
| M3 | Baselines | Not started | EvalSQL M1 harness is now committed, so this is unblocked once data exists |
| M4 | SFT run + ablations | Not started | Needs GPU run by human |
| M5 | DPO | Not started | Needs GPU run by human |
| M6 | Quantize, serve (vLLM), benchmark | Not started | Needs GPU |
| M7 | Publish (HF, model card, dashboard) | Not started | |
| M8 | Integrate into EvalSQL (optional) | Not started | |

## Metrics (measured only)
| Model | Exec accuracy | Split / sample size | Cost per 1,000 queries | p95 latency | Date |
|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct zero-shot | not yet measured | | | | |
| Large API model | not yet measured | | | | |
| SFT | not yet measured | | | | |
| SFT + DPO | not yet measured | | | | |
| SFT + DPO, AWQ 4-bit | not yet measured | | | | |

## Ablations
| Variant | Exec accuracy | Notes |
|---|---|---|
| not yet run | | |

## Decisions
- **Tracking:** plain JSON run records (`run_meta.json`, `log.jsonl`, `result.json`), no W&B. No account needed and the files can be committed as evidence. W&B stays available via `report_to`.
- **Shared prompt.** Prompt format `v1` is defined in `data/prompt.py`. `build_prompt()` is
  the only way to build a prompt, and SFT, DPO, baselines and serving must all use it. The
  schema is CREATE TABLE text rebuilt from SQLite metadata, with 3 example rows per table,
  plus BIRD evidence as a `Hint:` line.
- **Gold SQL is executed** with `t2s_eval.execute_sql`, and failing labels are dropped. The
  same executor is used as for evaluation, so there is no second scorer.
- **Leakage policy (conservative).** A training example is dropped if it shares any of the
  following with Spider dev/test or BIRD dev, across both datasets:
  - the db_id;
  - an identical or near-copy schema (Jaccard ≥ 0.6, or containment ≥ 0.8);
  - the normalized question.

  The written files are re-checked after writing. Missing eval databases stop the run.
- **Validation split.** 5% of training *databases* are held out (seed 0), so validation
  loss is measured on unseen schemas.
- **Length filter.** Prompts over 16,000 characters are dropped. Token-level limits are
  applied in M2 with the real Qwen tokenizer.

- **M2 loss mask.** Loss is on the SQL answer and end-of-turn token only. Masks are computed in `train/data.py` and passed to TRL as a `labels` column (TRL 1.16 uses it as is), because Qwen's chat template has no assistant markers for `assistant_only_loss`. Over-length examples are dropped, not cut.
- **M2 libraries.** Plain backend pinned and exercised on 2026-10-21: transformers 5.20.2, trl 1.16.0, peft 0.21.2, datasets 5.1.0, accelerate 1.15.0 (CPU torch). The smoke test found one incompatibility: transformers 5 removed `warmup_ratio`, so warmup is passed as an integer `warmup_steps`. `unsloth==2026.10.7` is pinned alone; its resolved dependency versions get recorded in `run_meta.json` and must be copied into `requirements-unsloth.txt` after the first GPU run.
- **Sibling repo name.** The evaluator repo is `EvalSQL` on GitHub, so paths now say `../EvalSQL/eval` (they said `../text2sql/eval`, which would not have installed).

## Blockers / needs from human
- **Dataset download.** The cloud environment blocks the hosts. Follow docs/DATA.md
  ("Download") to put Spider in `data/raw/spider/` and BIRD train and dev in
  `data/raw/bird/`. Then run:
  ```bash
  pip install -r requirements-data.txt
  python -m data.prepare --spider-dir data/raw/spider --bird-dir data/raw/bird --out data/processed
  ```
  Bring back `data/processed/stats.json`, which fills the sizes table in docs/DATA.md, and
  the first ~200 lines of `data/processed/dropped.jsonl`, to check the near-copy leakage
  drops. Results are **not yet measured**.
- **First GPU run (M2 check).** Open `notebooks/sft_colab.ipynb` on a Colab GPU, run the dry run and the smoke cell, and bring back the output. This is the first real test of the Unsloth backend and 4-bit loading. See docs/TRAINING.md.
- GPU runs (M4, M5, M6) must be executed by the human on Colab/Kaggle/rented GPU; Claude supplies scripts and exact commands.

## Log
- 2026-10-21: repo scaffolded with CLAUDE.md, subagents, hooks.
- 2026-10-20: M1. Built the data pipeline (schema serialization, shared prompt, cleaning, label execution check, dedup, eval-leakage removal, database-grouped split) and committed it (aa91ff9). The reviewer found keyword-named tables crashing the run, SQL string literals being altered, missing eval databases disabling a leakage check, and cross-source and near-copy schema gaps. All were fixed, with regression tests (c24cff5), and the re-review passed after one more fix for shadow-table matching. Tests: 35 passed (Python 3.11 and 3.13); ruff and mypy --strict clean.
- 2026-10-21: M2. Added `train/` (strict TOML config, tokenization with answer-only loss mask, Unsloth and plain backends, resume after disconnect, JSON run records), four configs plus a smoke config, a Colab notebook, and docs/TRAINING.md. Tests: 76 passed and 1 opt-in skipped in the default run (the 37 new ones cover config and data); `SMOKE=1` ran 3 real training steps of Qwen2.5-Coder-0.5B on CPU, 81 s with the model already downloaded. ruff and mypy --strict clean. Fixed the `../text2sql` path to `../EvalSQL`.
