"""Supervised fine-tuning of Qwen2.5-Coder with 4-bit QLoRA.

    python -m train.sft --config train/configs/sft_r16.toml
    python -m train.sft --config train/configs/sft_r16.toml --dry-run          # no model, no GPU
    python -m train.sft --config train/configs/smoke.toml                # tiny end-to-end check
    python -m train.sft --config train/configs/sft_r16.toml \
        --set run_name='"sft_n2000"' max_train_examples=2000

Everything a run produces goes to outputs/<run_name>/:
    run_meta.json    config, library versions, GPU, git commit, hashes of the data files
    data_stats.json  how many examples were kept or dropped, and token lengths
    log.jsonl        one line per logging step (loss, learning rate, eval loss)
    adapter/         the LoRA adapter and tokenizer
    result.json      final metrics, runtime, peak GPU memory

A run that is stopped (Colab disconnects) continues from its newest checkpoint when started again
with the same command.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from train import runlog
from train.config import ConfigError, SFTConfig, load_config, parse_overrides
from train.data import DataError, Tokenized, build_split, check_boundary, load_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
IGNORE_INDEX = -100
BOUNDARY_CHECK_SAMPLE = 25


def resolve_precision(cfg: SFTConfig) -> str:
    """'bf16', 'fp16' or 'fp32'. 'auto' picks the best the GPU supports, fp32 on CPU."""
    if cfg.precision != "auto":
        return cfg.precision
    import torch

    if torch.cuda.is_available():
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def to_columns(items: Sequence[Tokenized]) -> dict[str, list[list[int]]]:
    """Columns TRL treats as already tokenized: input_ids plus a labels column used as is."""
    return {
        "input_ids": [t.input_ids for t in items],
        "attention_mask": [[1] * len(t.input_ids) for t in items],
        "labels": [
            [i if m else IGNORE_INDEX for i, m in zip(t.input_ids, t.completion_mask, strict=True)]
            for t in items],
    }


def build_datasets(cfg: SFTConfig, tokenizer: Any, root: Path = REPO_ROOT
                   ) -> tuple[list[Tokenized], list[Tokenized], dict[str, Any]]:
    train_rows = load_jsonl(root / cfg.train_file)
    val_rows = load_jsonl(root / cfg.val_file)
    train, train_stats = build_split(tokenizer, train_rows, cfg.max_seq_length, "train",
                                     cfg.max_train_examples, cfg.seed)
    val, val_stats = build_split(tokenizer, val_rows, cfg.max_seq_length, "val",
                                 cfg.max_val_examples, cfg.seed)
    if not train:
        raise DataError("no training examples left after filtering; "
                        "check max_seq_length and the data")
    if not val:
        raise DataError("no validation examples left after filtering")

    sample = [r for r in train_rows if "messages" in r][:BOUNDARY_CHECK_SAMPLE]
    mismatches = sum(not check_boundary(tokenizer, r["messages"]) for r in sample)
    stats = {"train": train_stats.to_json(), "val": val_stats.to_json(),
             "boundary_check": {"checked": len(sample), "mismatches": mismatches}}
    if mismatches:
        print(f"warning: {mismatches}/{len(sample)} examples tokenize differently when the prompt "
              "and answer are split; check the chat template", file=sys.stderr)
    return train, val, stats


def steps_per_epoch(n_examples: int, cfg: SFTConfig) -> int:
    return max(1, n_examples // cfg.effective_batch_size)


def total_steps(n_examples: int, cfg: SFTConfig) -> int:
    if cfg.max_steps > 0:
        return cfg.max_steps
    return max(1, round(steps_per_epoch(n_examples, cfg) * cfg.num_train_epochs))


def latest_checkpoint(output_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for p in output_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Model loading (imports are lazy so --dry-run and the unit tests need no GPU stack)
# ---------------------------------------------------------------------------

def load_tokenizer(cfg: SFTConfig) -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_unsloth(cfg: SFTConfig) -> tuple[Any, Any]:
    # Unsloth patches transformers and TRL, so it has to be imported before them.
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name, max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.effective_lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none",
        use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
        random_state=cfg.seed)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_plain(cfg: SFTConfig, precision: str) -> tuple[Any, Any, Any]:
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    kwargs: dict[str, Any] = {"dtype": dtype}
    if cfg.load_in_4bit:
        if not torch.cuda.is_available():
            raise ConfigError("load_in_4bit needs a CUDA GPU; set load_in_4bit = false on CPU")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.effective_lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=list(cfg.target_modules),
                      bias="none", task_type="CAUSAL_LM")
    return model, load_tokenizer(cfg), lora


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_log_callback(path: Path) -> Any:
    from transformers import TrainerCallback

    class JsonLogCallback(TrainerCallback):  # type: ignore[misc]
        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> None:
            if logs:
                runlog.append_jsonl(path, {"step": state.global_step, "epoch": state.epoch, **logs})

    return JsonLogCallback()


def train(cfg: SFTConfig, root: Path = REPO_ROOT) -> dict[str, Any]:
    out = root / cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # The model comes first because Unsloth must be imported before TRL and transformers.
    precision = resolve_precision(cfg) if cfg.backend == "plain" else cfg.precision
    peft_config = None
    if cfg.backend == "unsloth":
        model, tokenizer = load_model_unsloth(cfg)
    else:
        model, tokenizer, peft_config = load_model_plain(cfg, precision)
    precision = resolve_precision(cfg)

    from datasets import Dataset
    from transformers import set_seed
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    set_seed(cfg.seed)
    train_items, val_items, data_stats = build_datasets(cfg, tokenizer, root)
    runlog.write_json(out / "data_stats.json", data_stats)
    n_steps = total_steps(len(train_items), cfg)

    meta = {
        "status": "started", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg.to_dict(), "precision": precision,
        "planned_steps": n_steps, "effective_batch_size": cfg.effective_batch_size,
        "environment": runlog.environment(root),
        "data_files": {cfg.train_file: runlog.sha256_file(root / cfg.train_file),
                       cfg.val_file: runlog.sha256_file(root / cfg.val_file),
                       "data/processed/stats.json":
                           runlog.sha256_file(root / "data" / "processed" / "stats.json")},
    }
    runlog.write_json(out / "run_meta.json", meta)

    args = TRLSFTConfig(
        output_dir=str(out), seed=cfg.seed, max_length=cfg.max_seq_length, packing=False,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        learning_rate=cfg.learning_rate, lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=round(cfg.warmup_ratio * n_steps), weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs, max_steps=cfg.max_steps, optim=cfg.optim,
        bf16=precision == "bf16", fp16=precision == "fp16",
        logging_steps=cfg.logging_steps, eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.save_steps, save_total_limit=cfg.save_total_limit,
        report_to=cfg.report_to, remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=Dataset.from_dict(to_columns(train_items)),
        eval_dataset=Dataset.from_dict(to_columns(val_items)),
        peft_config=peft_config,
        callbacks=[make_log_callback(out / "log.jsonl")],
    )

    resume = latest_checkpoint(out)
    if resume:
        print(f"resuming from {resume}")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    peak_gb: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except ImportError:
        pass

    result = {"status": "finished", "train_examples": len(train_items),
              "val_examples": len(val_items),
              "steps": trainer.state.global_step, "resumed_from": str(resume) if resume else None,
              "wall_seconds": round(time.time() - t0, 1), "train_metrics": train_result.metrics,
              "eval_metrics": eval_metrics, "peak_gpu_memory_gb": peak_gb}
    runlog.write_json(out / "result.json", result)
    meta["status"] = "finished"
    runlog.write_json(out / "run_meta.json", meta)
    return result


def dry_run(cfg: SFTConfig, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load data and the tokenizer only. Shows what a real run would train on."""
    tokenizer = load_tokenizer(cfg)
    train_items, _, stats = build_datasets(cfg, tokenizer, root)
    n_steps = total_steps(len(train_items), cfg)
    return {"run_name": cfg.run_name, "effective_batch_size": cfg.effective_batch_size,
            "planned_steps": n_steps, "data": stats}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="TOML file in train/configs/")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override config values; values are TOML, so strings need quotes")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data and tokenizer, print stats, stop")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config, parse_overrides(args.set))
        report = dry_run(cfg) if args.dry_run else train(cfg)
    except (ConfigError, DataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
