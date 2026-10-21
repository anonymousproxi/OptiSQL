"""Regenerate notebooks/sft_colab.ipynb. Run: python notebooks/build_notebook.py"""

import json
from pathlib import Path

OUT = Path(__file__).with_name("sft_colab.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# OptiSQL: SFT run (Colab / Kaggle)

Runs `train/sft.py` on a GPU. Do the cells in order. Nothing here is a result until the numbers
come back in `result.json` and are copied into `PROGRESS.md`.

You need first, on your own computer: `data/processed/` produced by `python -m data.prepare`
(see `docs/DATA.md`). It is not in git, so upload it to Google Drive (or a Kaggle dataset).
"""),
    md("## 1. GPU and storage"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount("/content/drive")
# Where data/processed/ lives on your Drive, and where run outputs should be kept (survives disconnects).
DATA_DIR = "/content/drive/MyDrive/optisql/processed"
OUT_ROOT = "/content/drive/MyDrive/optisql/outputs"
"""),
    md("## 2. Code and pinned libraries"),
    code("""
!git clone https://github.com/anonymousproxi/EvalSQL.git /content/EvalSQL
!git clone https://github.com/anonymousproxi/OptiSQL.git /content/OptiSQL
%cd /content/OptiSQL
!pip install -q -r requirements-unsloth.txt
!pip install -q -e ../EvalSQL/eval
!mkdir -p data/processed && cp {DATA_DIR}/*.jsonl {DATA_DIR}/stats.json data/processed/
!ls -la data/processed
"""),
    md("## 3. Dry run: what will the run train on? (no model, no GPU memory)"),
    code("""
!python -m train.sft --config train/configs/sft_r16.toml --dry-run
"""),
    md("""
Check `kept` against `loaded`. A large `dropped_too_long` means `max_seq_length` is too small.
`boundary_check.mismatches` must be 0.
"""),
    md("## 4. Smoke test (about 5 minutes). Do this before the long run."),
    code("""
!python -m train.sft --config train/configs/smoke.toml --set output_root=\\"{OUT_ROOT}\\"
"""),
    md("## 5. The real run"),
    code("""
!python -m train.sft --config train/configs/sft_r16.toml --set output_root=\\"{OUT_ROOT}\\"
"""),
    md("""
If Colab disconnects, reconnect, run sections 1 and 2 again, then run this cell again. The run
continues from its newest checkpoint on Drive.

Ablations (M4): swap the config, for example `train/configs/sft_r8.toml`, or override values:
`--set run_name=\\"sft_n2000\\" max_train_examples=2000`.
"""),
    md("## 6. Bring these back"),
    code("""
import json, glob
for run in sorted(glob.glob(f"{OUT_ROOT}/*/result.json")):
    print(run); print(json.dumps(json.load(open(run)), indent=1)[:1500])
print("Copy these files to the repo's outputs/<run_name>/ folder (checkpoints are not needed):")
for f in sorted(glob.glob(f"{OUT_ROOT}/*/*.json*")): print(" ", f)
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf8")
print("wrote", OUT)
