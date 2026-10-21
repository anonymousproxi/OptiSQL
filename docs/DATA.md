# Data

## Sources and licenses

| Dataset | Used for | License | How it was checked |
|---|---|---|---|
| Spider 1.0 (Yu et al., 2018) | SFT train: `train_spider.json`, `train_others.json`. Leakage reference only: `dev.json`, `test.json` | Dataset: CC BY-SA 4.0 as stated on https://yale-lily.github.io/spider. **Not yet verified**: that page was blocked from the build environment, so confirm it when you download. The GitHub repo code (taoyds/spider) is Apache-2.0. | Repo `LICENSE` file read on 2026-10-20 |
| BIRD (Li et al., 2023) | SFT train: `train/train.json`. Leakage reference only: `dev/dev.json` | CC BY-SA 4.0 | Official README (AlibabaResearch/DAMO-ConvAI, `bird/README.md`) read on 2026-10-20: "now we change the license to CC BY-SA 4.0" |
| Qwen2.5-Coder-7B-Instruct | Base model | Check the model card's license before publishing | To check at M7 |

What this means for the project:

- **ShareAlike.** If we publish any processed dataset (for example the SFT JSONL files), it
  must also be CC BY-SA 4.0, with attribution to Spider and BIRD. Whether ShareAlike also
  applies to fine-tuned weights is legally unsettled. The model card (M7) will name the
  training data and its license and will not claim otherwise.
- **No redistribution of raw data in git.** `data/raw/` and `data/processed/` are
  git-ignored, and `data/prepare.py` regenerates `processed/` from the raw downloads.
- **Attribution.** Cite both papers in the README and the model card.
- **Upstream sources.** BIRD's databases were built from public sources (for example Kaggle
  datasets), which may have terms of their own. Review these before publishing anything
  that redistributes database contents, such as the example rows inside the prompts.

## Download (manual)

The build environment cannot reach the dataset hosts, so you download and unpack these
yourself. Paths are relative to `sql-finetune/`.

Requirements: Python 3.11 or newer, and the `EvalSQL` repo cloned next to this one
(`../EvalSQL`), because `requirements-data.txt` installs `t2s_eval` from there.

1. **Spider.** From https://yale-lily.github.io/spider, download the Spider dataset zip and
   unpack it so these paths exist. Newer zips unpack to a `spider_data/` folder; rename it
   to `spider`.
   - `data/raw/spider/train_spider.json`
   - `data/raw/spider/train_others.json`
   - `data/raw/spider/dev.json`
   - `data/raw/spider/database/<db_id>/<db_id>.sqlite`
   - Newer releases also include `test.json` and `test_database/`. If present, they are
     used automatically as extra leakage references.
2. **BIRD.** From https://bird-bench.github.io/, download the train and dev zips. They contain
   nested `*_databases.zip` archives, which you also need to unpack. You should end up with:
   - `data/raw/bird/train/train.json`
   - `data/raw/bird/train/train_databases/<db_id>/<db_id>.sqlite`
   - `data/raw/bird/dev/dev.json`
   - `data/raw/bird/dev/dev_databases/<db_id>/<db_id>.sqlite`

   If your dev folder has a dated name such as `dev_20240627`, rename it to `dev`. The
   dev databases must be unpacked too, even though they are only used for checking. The
   leakage checks compare schemas, so the run stops if any eval database file is missing.
3. Run the pipeline:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-data.txt        # installs ../EvalSQL/eval (t2s_eval)
   python -m data.prepare --spider-dir data/raw/spider --bird-dir data/raw/bird --out data/processed
   ```
   Executing every BIRD gold query can take several minutes, because some BIRD databases
   are large.

## Pipeline (`data/prepare.py`)

1. **Load** each train file with `t2s_eval.load_examples`, the same loader the evaluator uses.
2. **Clean.**
   - Collapse whitespace in questions and evidence.
   - In SQL, collapse whitespace *outside* quotes only, so string literals and quoted
     identifiers stay exactly as written. Strip any trailing `;`.
   - Drop empty rows.
3. **Check labels.** Execute every gold SQL query read-only with `t2s_eval.execute_sql`.
   Drop any that fail (`gold_sql_error`), time out after 30 s by default (`gold_timeout`),
   or return more than 100,000 rows (`gold_too_many_rows`). Training on broken labels teaches broken
   SQL. You can turn this step off with `--skip-exec`.
4. **Remove duplicates**, within each source and database:
   - An exact duplicate (same question, evidence and SQL) is dropped.
   - When the same question has two different SQL labels, only the first is kept and the
     rest are dropped as `conflicting_duplicate`.
   - Questions are matched after Unicode NFKC normalization, case folding, and removal of
     punctuation, so `What’s` matches `what's`.
5. **Remove eval overlap.** The eval splits are Spider dev, Spider test (if present) and BIRD
   dev. Spider and BIRD are checked against each other as well as against themselves. A
   training example is dropped if:
   - its `db_id` matches any eval database's, in either dataset (`leak_eval_db`). For
     example, Spider's `formula_1` is dropped because BIRD dev has a `formula_1`.
   - its database has exactly the same table and column names as an eval database, under
     any name (`leak_eval_schema`).
   - its database is a near-copy of an eval database (`leak_eval_schema_similar`), meaning
     either:
     - the `table.column` sets have Jaccard similarity ≥ 0.6; or
     - at least 80% of the smaller schema, and at least 4 columns, appear in the larger one.

     This catches copies that are renamed or have an extra table. `dropped.jsonl` names the
     matching eval database and both scores.
   - its normalized question equals any eval question (`leak_eval_question`). This is
     conservative, because a generic question on a different database is dropped too.

   After writing, the script re-reads `sft_train.jsonl` and `sft_val.jsonl` and re-runs
   every check. If anything overlaps, it deletes the files and stops. If any eval database
   file is missing, the run stops before starting, because the schema checks could not
   run. `--allow-missing-eval-dbs` overrides this, and the weaker check is then recorded
   in `stats.json`.
6. **Build prompts** with `data.prompt.build_prompt(db_path, question, evidence)`. This is
   the one shared format for SFT, DPO, baselines and serving (prompt version `v1`), and it
   fixes the number of example rows, so callers cannot drift from it. It contains:
   - a system instruction;
   - the schema as CREATE TABLE text rebuilt from PRAGMA metadata:
     - primary and foreign keys, with composite keys grouped;
     - names quoted when they are SQL keywords or contain unusual characters;
     - views with their definitions;
     - FTS shadow tables left out;
     - 3 example rows per table, written as SQL literals and cut to 40 characters;
   - the BIRD evidence as a `Hint:` line;
   - the question.

   The assistant turn is the bare SQL. Examples whose full chat text is over 16,000
   characters are dropped (`too_long`). Token-level truncation is handled in `train/sft.py`
   with the real tokenizer.
7. **Split.** 5% of the training *databases* are held out as `sft_val.jsonl` (seed 0). Whole
   databases are held out, so validation loss is measured on unseen schemas.

Outputs in `data/processed/`:

- `sft_train.jsonl` and `sft_val.jsonl`: `{id, source, db_id, question, evidence, sql, messages}`.
- `dropped.jsonl`: every dropped example, with its reason.
- `stats.json`: the counts for each stage, prompt-length stats, the settings, the
  `t2s_eval` version, which eval databases were checked or missing, and the SHA-256 of
  every input JSON file, for reproducibility. The databases themselves are not hashed,
  because BIRD's are several GB.

## Sizes

**Not yet measured.** The datasets have not been downloaded here yet. After running the
pipeline, copy the numbers from `data/processed/stats.json` into this table. Do not use
published dataset sizes instead.

| | Spider | BIRD |
|---|---|---|
| Loaded (train files) | not yet measured | not yet measured |
| Dropped: gold SQL fails | not yet measured | not yet measured |
| Dropped: duplicates (exact / conflicting) | not yet measured | not yet measured |
| Dropped: eval overlap (db / schema / near-copy schema / question) | not yet measured | not yet measured |
| Dropped: too long | not yet measured | not yet measured |
| Final train / val | not yet measured | not yet measured |
| Training databases | not yet measured | not yet measured |
