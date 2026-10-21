from __future__ import annotations

from pathlib import Path

import pytest

from train.config import (
    ConfigError,
    SFTConfig,
    from_dict,
    load_config,
    parse_overrides,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "train" / "configs"


def test_every_shipped_config_is_valid() -> None:
    files = sorted(CONFIG_DIR.glob("*.toml"))
    assert {f.stem for f in files} >= {"smoke", "sft_r8", "sft_r16", "sft_r32"}
    for f in files:
        cfg = load_config(f)
        assert cfg.run_name  # validation ran


def test_ablation_configs_differ_only_where_intended() -> None:
    base = load_config(CONFIG_DIR / "sft_r16.toml").to_dict()
    expected = {
        "sft_r8": {"run_name": "sft_r8", "lora_r": 8},
        "sft_r32": {"run_name": "sft_r32", "lora_r": 32},
        "sft_r16_lr1e-4": {"run_name": "sft_r16_lr1e-4", "learning_rate": 1e-4},
    }
    for stem, changes in expected.items():
        other = load_config(CONFIG_DIR / f"{stem}.toml").to_dict()
        diff = {k: other[k] for k in other if other[k] != base[k]}
        assert diff == changes, stem


def test_defaults_and_derived_values() -> None:
    cfg = from_dict({"run_name": "x", "lora_r": 8})
    assert cfg.effective_lora_alpha == 16
    assert from_dict({"run_name": "x", "lora_r": 8, "lora_alpha": 8}).effective_lora_alpha == 8
    assert cfg.effective_batch_size == 16
    assert cfg.output_dir == Path("outputs") / "x"
    assert cfg.to_dict()["target_modules"][0] == "q_proj"


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown config keys: lora_rank"):
        from_dict({"run_name": "x", "lora_rank": 8})


@pytest.mark.parametrize("bad", [
    {"lora_r": 0},
    {"learning_rate": 0},
    {"learning_rate": -1e-4},
    {"lora_dropout": 1.0},
    {"warmup_ratio": 1.5},
    {"max_seq_length": 0},
    {"max_train_examples": 0},
    {"backend": "tpu"},
    {"precision": "fp8"},
    {"target_modules": []},
    {"num_train_epochs": 0},
    {"eval_steps": 7, "save_steps": 10},
    {"backend": "plain", "optim": "adamw_8bit"},
    {"run_name": "a/b"},
    {"run_name": "has space"},
    {"run_name": ""},
])
def test_invalid_values_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        from_dict({"run_name": "x", **bad})


def test_missing_run_name_and_missing_file() -> None:
    with pytest.raises(ConfigError, match="run_name"):
        from_dict({})
    with pytest.raises(ConfigError, match="not found"):
        load_config(CONFIG_DIR / "nope.toml")


def test_max_steps_can_replace_epochs() -> None:
    cfg = from_dict({"run_name": "x", "num_train_epochs": 0.0001, "max_steps": 5})
    assert cfg.max_steps == 5


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    f = tmp_path / "bad.toml"
    f.write_text("run_name = ")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(f)


def test_overrides_keep_types_and_win(tmp_path: Path) -> None:
    f = tmp_path / "c.toml"
    f.write_text('run_name = "a"\nlora_r = 16\n')
    over = parse_overrides(["lora_r=8", "run_name=\"b\"", "load_in_4bit=false",
                            "learning_rate=1e-4", "model_name=Qwen/Qwen2.5-Coder-0.5B-Instruct"])
    assert over == {"lora_r": 8, "run_name": "b", "load_in_4bit": False, "learning_rate": 1e-4,
                    "model_name": "Qwen/Qwen2.5-Coder-0.5B-Instruct"}
    cfg = load_config(f, over)
    assert (cfg.run_name, cfg.lora_r, cfg.load_in_4bit) == ("b", 8, False)
    with pytest.raises(ConfigError, match="key=value"):
        parse_overrides(["oops"])


def test_config_is_immutable() -> None:
    cfg: SFTConfig = from_dict({"run_name": "x"})
    with pytest.raises(AttributeError):
        cfg.lora_r = 99  # type: ignore[misc]
