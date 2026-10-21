"""Training configuration, loaded from TOML and validated strictly.

Unknown keys and out-of-range values stop the run before any model is downloaded, so a typo in
an ablation config cannot silently fall back to a default.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj")
BACKENDS = ("unsloth", "plain")


class ConfigError(ValueError):
    """The config file is malformed or has a value that cannot work."""


@dataclass(frozen=True)
class SFTConfig:
    run_name: str
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "unsloth": 4-bit QLoRA on a CUDA GPU (the real runs). "plain": transformers + peft, any
    # device, used for the CPU smoke test and as a fallback.
    backend: str = "unsloth"
    load_in_4bit: bool = True

    train_file: str = "data/processed/sft_train.jsonl"
    val_file: str = "data/processed/sft_val.jsonl"
    # Outputs go to <output_root>/<run_name>.
    output_root: str = "outputs"

    # Longer examples are dropped, never cut: a cut SQL target teaches the model to stop early.
    max_seq_length: int = 4096
    # Data-size ablation: keep this many training examples, chosen by a seeded hash of the id so
    # a smaller subset is always contained in a larger one.
    max_train_examples: int | None = None
    max_val_examples: int | None = 500
    seed: int = 0

    lora_r: int = 16
    lora_alpha: int | None = None  # defaults to 2 * lora_r
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_train_epochs: float = 2.0
    max_steps: int = -1  # overrides epochs when positive
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"
    precision: str = "auto"  # "auto", "bf16", "fp16" or "fp32"

    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    report_to: str = "none"  # plain JSON logs are always written; "wandb" adds W&B

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name

    @property
    def effective_lora_alpha(self) -> int:
        return self.lora_alpha if self.lora_alpha is not None else 2 * self.lora_r

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


_POSITIVE_INTS = ("max_seq_length", "lora_r", "per_device_train_batch_size",
                  "per_device_eval_batch_size", "gradient_accumulation_steps",
                  "logging_steps", "eval_steps", "save_steps", "save_total_limit")


def _validate(cfg: SFTConfig) -> None:
    def bad(msg: str) -> ConfigError:
        return ConfigError(f"{cfg.run_name}: {msg}")

    if not cfg.run_name or any(c in cfg.run_name for c in "/\\ "):
        raise ConfigError(
            f"run_name must be a non-empty name without slashes or spaces: {cfg.run_name!r}")
    if cfg.backend not in BACKENDS:
        raise bad(f"backend must be one of {BACKENDS}, got {cfg.backend!r}")
    if cfg.backend == "plain" and cfg.optim == "adamw_8bit":
        raise bad("optim 'adamw_8bit' needs the unsloth backend; use 'adamw_torch' with plain")
    for name in _POSITIVE_INTS:
        if getattr(cfg, name) <= 0:
            raise bad(f"{name} must be positive, got {getattr(cfg, name)}")
    for name in ("max_train_examples", "max_val_examples"):
        v = getattr(cfg, name)
        if v is not None and v <= 0:
            raise bad(f"{name} must be positive or unset, got {v}")
    if cfg.lora_alpha is not None and cfg.lora_alpha <= 0:
        raise bad("lora_alpha must be positive")
    if not 0.0 <= cfg.lora_dropout < 1.0:
        raise bad("lora_dropout must be in [0, 1)")
    if not 0.0 <= cfg.warmup_ratio < 1.0:
        raise bad("warmup_ratio must be in [0, 1)")
    if cfg.learning_rate <= 0:
        raise bad("learning_rate must be positive")
    if cfg.num_train_epochs <= 0 and cfg.max_steps <= 0:
        raise bad("set num_train_epochs or max_steps")
    if cfg.precision not in ("auto", "bf16", "fp16", "fp32"):
        raise bad(f"precision must be auto, bf16, fp16 or fp32, got {cfg.precision!r}")
    if not cfg.target_modules:
        raise bad("target_modules must not be empty")
    if cfg.save_steps % cfg.eval_steps != 0:
        # Required by load_best_model_at_end style checkpoint logic and keeps eval and save aligned.
        raise bad("save_steps must be a multiple of eval_steps")


def from_dict(raw: dict[str, Any]) -> SFTConfig:
    known = {f.name for f in fields(SFTConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown config keys: {', '.join(unknown)}")
    if "run_name" not in raw:
        raise ConfigError("config needs a run_name")
    values = dict(raw)
    if "target_modules" in values:
        values["target_modules"] = tuple(values["target_modules"])
    try:
        cfg = SFTConfig(**values)
    except TypeError as e:  # wrong argument shape
        raise ConfigError(str(e)) from e
    _validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SFTConfig:
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {p}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{p}: invalid TOML: {e}") from e
    raw.update(overrides or {})
    return from_dict(raw)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """`key=value` strings from the command line. Values are read as TOML, so numbers and
    booleans keep their type; anything that is not valid TOML is taken as a string."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override must look like key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        try:
            out[key] = tomllib.loads(f"v = {value}")["v"]
        except tomllib.TOMLDecodeError:
            out[key] = value
    return out
