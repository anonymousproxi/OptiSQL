from __future__ import annotations

from pathlib import Path

from train.config import from_dict
from train.data import Tokenized
from train.sft import IGNORE_INDEX, latest_checkpoint, to_columns, total_steps


def test_columns_mask_the_prompt_with_ignore_index() -> None:
    items = [Tokenized([10, 11, 12, 13], [0, 0, 1, 1]), Tokenized([20, 21], [0, 1])]
    cols = to_columns(items)
    assert cols["input_ids"] == [[10, 11, 12, 13], [20, 21]]
    assert cols["labels"] == [[IGNORE_INDEX, IGNORE_INDEX, 12, 13], [IGNORE_INDEX, 21]]
    assert cols["attention_mask"] == [[1, 1, 1, 1], [1, 1]]


def test_total_steps_follows_epochs_and_effective_batch() -> None:
    cfg = from_dict({"run_name": "x", "per_device_train_batch_size": 2,
                     "gradient_accumulation_steps": 8, "num_train_epochs": 2.0})
    assert cfg.effective_batch_size == 16
    assert total_steps(1600, cfg) == 200
    assert total_steps(5, cfg) == 2  # fewer examples than one batch still takes a step per epoch


def test_max_steps_overrides_epochs() -> None:
    cfg = from_dict({"run_name": "x", "max_steps": 7})
    assert total_steps(100_000, cfg) == 7


def test_latest_checkpoint_picks_the_highest_step(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
    for name in ("checkpoint-200", "checkpoint-1000", "checkpoint-30", "checkpoint-x", "other"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-5000").write_text("a file, not a checkpoint directory")
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-1000"
