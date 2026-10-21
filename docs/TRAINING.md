# Training (M2)

Supervised fine-tuning of `Qwen/Qwen2.5-Coder-7B-Instruct` with 4-bit QLoRA, using the data
from `data/prepare.py`. The scripts are written and tested; **no real training run has been done
yet**, so there are no accuracy or memory numbers. Everything marked "not yet measured" below
needs a GPU run by you.

## What the code does

| File | Role |
|---|---|
| `train/config.py` | Strict TOML config. Unknown keys and impossible values stop the run before any download. |
| `train/data.py` | Chat JSONL to token ids. Loss mask: SQL answer and end-of-turn token only. Too-long examples are dropped, never cut. Seeded, nested data-size subsets. |
| `train/sft.py` | The trainer: Unsloth backend for real runs, plain transformers + peft backend for CPU and the smoke test. Resumes after a disconnect. |
| `train/runlog.py` | Plain JSON run records. |
| `train/configs/` | `sft_r16` (main), `sft_r8`, `sft_r32`, `sft_r16_lr1e-4` (ablations), `smoke`. |
| `notebooks/sft_colab.ipynb` | Colab notebook that runs the steps below. |

Each run writes `outputs/<run_name>/`: `run_meta.json` (config, library versions, GPU, git commit,
SHA-256 of the data files), `data_stats.json`, `log.jsonl` (loss every `logging_steps`),
`result.json` (final metrics, wall time, peak GPU memory), and `adapter/`.

## Decisions

- **Loss on the answer only.** The prompt (schema, hint, question) is masked with `-100`. TRL's
  `assistant_only_loss` needs template markers that Qwen's template lacks, so masks are computed
  in `train/data.py` and passed as a `labels` column, which TRL 1.16 uses as is.
- **Prompt and answer are tokenized separately** and checked against joint tokenization on the
  first 25 examples of every run (`boundary_check.mismatches` must be 0).
- **Over-length examples are dropped.** Default `max_seq_length = 4096`. M1 already drops
  prompts over 16,000 characters; `data_stats.json` shows how many more go at the token level.
- **Tracking:** plain JSON files, not Weights & Biases (works with no account, and the files can
  be committed as evidence). Set `report_to = "wandb"` to add W&B.
- **Effective batch size 16** (2 x 8 accumulation), learning rate 2e-4, cosine schedule, 2 epochs,
  LoRA on all attention and MLP projections, alpha = 2 x rank. These are common QLoRA defaults,
  not tuned values; M4 ablates rank and learning rate.

## What was tested here

| Check | Result |
|---|---|
| Config and data unit tests (offline) | see PROGRESS.md |
| `SMOKE=1 pytest tests/test_train_smoke.py`: 3 real training steps of Qwen2.5-Coder-0.5B with the plain backend (CPU), then eval, checkpoint and adapter save | see PROGRESS.md |
| Unsloth backend (`load_model_unsloth`) | **not yet run**: it needs a CUDA GPU. The first Colab smoke run is the test. |
| 4-bit loading (`load_in_4bit`) | **not yet run**, same reason |
| GPU memory for the 7B run at `max_seq_length = 4096`, batch 2 | **not yet measured** |
| Time per epoch | **not yet measured** |

## How to run

You need `data/processed/sft_train.jsonl`, `sft_val.jsonl` and `stats.json` from
[DATA.md](DATA.md), and a GPU with 16 to 24 GB (Colab, Kaggle, or rented).

1. Open `notebooks/sft_colab.ipynb` in Colab and pick a GPU runtime.
2. Put `data/processed/` on Google Drive and set `DATA_DIR` in the first cell.
3. Run the cells in order: dry run, then the smoke test (about 5 minutes), then the real run.
4. Bring back `outputs/<run_name>/result.json`, `run_meta.json`, `data_stats.json` and
   `log.jsonl` for every run (not the checkpoints).

Without the notebook:

```bash
pip install -r requirements-unsloth.txt && pip install -e ../EvalSQL/eval
python -m train.sft --config train/configs/sft_r16.toml --dry-run   # what will it train on?
python -m train.sft --config train/configs/smoke.toml               # tiny end-to-end check
python -m train.sft --config train/configs/sft_r16.toml             # the real run
```

On Kaggle: add `data/processed/` as a dataset, enable a GPU (T4 x2 or P100), and replace the Drive
paths in the notebook with `/kaggle/input/...` and `/kaggle/working/...`. A 7B QLoRA run at
4096 tokens may not fit a 16 GB card; if it runs out of memory, lower
`--set max_seq_length=2048 per_device_train_batch_size=1 gradient_accumulation_steps=16` and say so
in the results, since it changes what is being compared.

## Ablations (M4)

Run each against the same validation set and evaluate with `t2s_eval` (M3 and M4 wrappers).

| Question | How |
|---|---|
| LoRA rank | `sft_r8.toml`, `sft_r16.toml`, `sft_r32.toml` |
| Learning rate | `sft_r16.toml` (2e-4) vs `sft_r16_lr1e-4.toml` |
| Data size | `--set run_name=\"sft_n1000\" max_train_examples=1000`, then 2000, 4000, and the full set. Smaller subsets sit inside larger ones, so the comparison is clean. |

## Library versions

`requirements-train.txt` pins the versions exercised by the smoke test. `requirements-unsloth.txt`
pins only `unsloth`, which chooses its own `trl`, `transformers` and `peft`; the versions it
resolves are recorded in `run_meta.json` and should be copied into that file after the first GPU run.
