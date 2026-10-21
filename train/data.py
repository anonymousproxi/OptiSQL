"""Turn the M1 chat JSONL into token ids with a loss mask.

The loss is computed on the assistant turn only (the SQL and the end-of-turn token), never on
the schema, hint or question. Examples longer than `max_seq_length` tokens are dropped rather
than cut, because a cut target teaches the model to stop mid-query.

Everything here takes any tokenizer with `apply_chat_template` and a call interface, so tests
run offline with a fake one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Tokenizer(Protocol):
    def apply_chat_template(self, conversation: Any, *, tokenize: bool = ...,
                            add_generation_prompt: bool = ..., **kwargs: Any) -> Any: ...

    def __call__(self, text: str, *, add_special_tokens: bool = ..., **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Tokenized:
    input_ids: list[int]
    # 1 on the tokens the loss is computed on, 0 elsewhere.
    completion_mask: list[int]


@dataclass
class DataStats:
    split: str
    loaded: int = 0
    selected: int = 0
    dropped_too_long: int = 0
    dropped_bad_shape: int = 0
    kept: int = 0
    tokens_total: int = 0
    completion_tokens_total: int = 0
    lengths: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        ls = sorted(self.lengths)

        def pct(p: float) -> int:
            return ls[min(len(ls) - 1, int(p * len(ls)))] if ls else 0

        return {"split": self.split, "loaded": self.loaded, "selected": self.selected,
                "dropped_too_long": self.dropped_too_long,
                "dropped_bad_shape": self.dropped_bad_shape, "kept": self.kept,
                "tokens_total": self.tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "length_tokens": {"p50": pct(0.5), "p95": pct(0.95), "max": ls[-1] if ls else 0}}


class DataError(ValueError):
    """The training file is not in the shape data/prepare.py writes."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataError(f"{path} not found. Run data/prepare.py first (see docs/DATA.md).")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{n}: invalid JSON: {e}") from e
    return rows


def select_subset(rows: Sequence[dict[str, Any]], limit: int | None,
                  seed: int) -> list[dict[str, Any]]:
    """The first `limit` rows when ordered by a seeded hash of the id.

    The order depends only on (seed, id), so a smaller subset is always contained in a larger
    one, which keeps data-size ablations comparable.
    """
    if limit is None or limit >= len(rows):
        return list(rows)

    def key(r: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()

    return sorted(rows, key=key)[:limit]


def _ids(tokenizer: Tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if not isinstance(encoded, list) else encoded
    return [int(i) for i in ids]


def split_prompt_and_completion(tokenizer: Tokenizer, messages: Sequence[dict[str, str]]
                                ) -> tuple[str, str]:
    """Chat-template text for the prompt (with the assistant header) and for the SQL answer."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise DataError("messages must end with an assistant turn")
    prompt = tokenizer.apply_chat_template(list(messages[:-1]), tokenize=False,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(list(messages), tokenize=False)
    if not isinstance(prompt, str) or not isinstance(full, str):
        raise DataError("tokenizer.apply_chat_template(tokenize=False) must return a string")
    if not full.startswith(prompt):
        raise DataError("chat template does not extend the prompt with the answer; "
                        "cannot find the assistant span")
    # The template may add a newline after the end-of-turn token; the model never needs to emit it.
    return prompt, full[len(prompt):].rstrip("\n")


def tokenize_messages(tokenizer: Tokenizer, messages: Sequence[dict[str, str]]) -> Tokenized:
    prompt, completion = split_prompt_and_completion(tokenizer, messages)
    prompt_ids = _ids(tokenizer, prompt)
    completion_ids = _ids(tokenizer, completion)
    if not completion_ids:
        raise DataError("empty assistant turn")
    return Tokenized(prompt_ids + completion_ids,
                     [0] * len(prompt_ids) + [1] * len(completion_ids))


def check_boundary(tokenizer: Tokenizer, messages: Sequence[dict[str, str]]) -> bool:
    """True when tokenizing prompt and answer separately equals tokenizing them together,
    which is what the model sees at serving time."""
    prompt, completion = split_prompt_and_completion(tokenizer, messages)
    separate = _ids(tokenizer, prompt) + _ids(tokenizer, completion)
    return separate == _ids(tokenizer, prompt + completion)


def build_split(tokenizer: Tokenizer, rows: Sequence[dict[str, Any]], max_len: int, split: str,
                limit: int | None = None, seed: int = 0) -> tuple[list[Tokenized], DataStats]:
    stats = DataStats(split=split, loaded=len(rows))
    chosen = select_subset(rows, limit, seed)
    stats.selected = len(chosen)
    out: list[Tokenized] = []
    for r in chosen:
        try:
            tok = tokenize_messages(tokenizer, r["messages"])
        except (KeyError, DataError):
            stats.dropped_bad_shape += 1
            continue
        if len(tok.input_ids) > max_len:
            stats.dropped_too_long += 1
            continue
        out.append(tok)
        stats.lengths.append(len(tok.input_ids))
        stats.tokens_total += len(tok.input_ids)
        stats.completion_tokens_total += sum(tok.completion_mask)
    stats.kept = len(out)
    return out, stats
