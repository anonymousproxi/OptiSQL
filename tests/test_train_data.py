from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from train.data import (
    DataError,
    build_split,
    check_boundary,
    load_jsonl,
    select_subset,
    split_prompt_and_completion,
    tokenize_messages,
)


class FakeTokenizer:
    """ChatML like Qwen's, one token per character, so masks can be checked by eye."""

    def apply_chat_template(self, conversation: list[dict[str, str]], *, tokenize: bool = True,
                            add_generation_prompt: bool = False, **_: Any) -> str:
        assert tokenize is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conversation)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text: str, *, add_special_tokens: bool = True,
                 **_: Any) -> dict[str, list[int]]:
        assert add_special_tokens is False  # the template already carries every special token
        return {"input_ids": [ord(c) for c in text]}


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


def messages(sql: str = "SELECT 1", question: str = "q") -> list[dict[str, str]]:
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": f"Question: {question}"},
            {"role": "assistant", "content": sql}]


def row(i: int, sql: str = "SELECT 1", question: str = "q") -> dict[str, Any]:
    return {"id": f"r{i}", "messages": messages(sql, question)}


def test_loss_mask_covers_only_the_assistant_answer() -> None:
    tok = tokenize_messages(FakeTokenizer(), messages("SELECT name FROM t"))
    trained = decode([i for i, m in zip(tok.input_ids, tok.completion_mask, strict=True) if m])
    assert trained == "SELECT name FROM t<|im_end|>"
    pairs = list(zip(tok.input_ids, tok.completion_mask, strict=True))
    untrained = decode([i for i, m in pairs if not m])
    assert untrained.endswith("<|im_start|>assistant\n")
    assert "SELECT" not in untrained
    assert len(tok.input_ids) == len(tok.completion_mask)


def test_completion_has_no_trailing_newline_but_keeps_end_of_turn() -> None:
    _, completion = split_prompt_and_completion(FakeTokenizer(), messages("SELECT 1"))
    assert completion == "SELECT 1<|im_end|>"


def test_boundary_tokenization_matches_joint_tokenization() -> None:
    assert check_boundary(FakeTokenizer(), messages())


def test_messages_must_end_with_an_assistant_turn() -> None:
    with pytest.raises(DataError, match="assistant"):
        tokenize_messages(FakeTokenizer(), messages()[:2])
    with pytest.raises(DataError, match="assistant"):
        tokenize_messages(FakeTokenizer(), messages()[2:])


def test_template_that_rewrites_the_prompt_is_rejected() -> None:
    class Rewriting(FakeTokenizer):
        def apply_chat_template(self, conversation: list[dict[str, str]], *, tokenize: bool = True,
                                add_generation_prompt: bool = False, **kw: Any) -> str:
            text = super().apply_chat_template(conversation, tokenize=tokenize,
                                               add_generation_prompt=add_generation_prompt, **kw)
            return text if add_generation_prompt else text.replace("sys", "SYS")

    with pytest.raises(DataError, match="does not extend"):
        tokenize_messages(Rewriting(), messages())


def test_empty_answer_is_rejected() -> None:
    class Dropping(FakeTokenizer):
        def apply_chat_template(self, conversation: list[dict[str, str]], *, tokenize: bool = True,
                                add_generation_prompt: bool = False, **kw: Any) -> str:
            text = super().apply_chat_template(conversation, tokenize=tokenize,
                                               add_generation_prompt=add_generation_prompt, **kw)
            return text.replace("<|im_end|>\n", "") if not add_generation_prompt else text

    with pytest.raises(DataError):
        tokenize_messages(Dropping(), [{"role": "user", "content": "q"},
                                       {"role": "assistant", "content": ""}])


def test_too_long_examples_are_dropped_not_cut() -> None:
    rows = [row(1), row(2, sql="SELECT " + "x" * 500)]
    out, stats = build_split(FakeTokenizer(), rows, max_len=200, split="train")
    assert stats.loaded == 2 and stats.kept == 1 and stats.dropped_too_long == 1
    assert len(out) == 1
    assert decode(out[0].input_ids).endswith("SELECT 1<|im_end|>")  # intact, not truncated


def test_bad_rows_are_counted_not_fatal() -> None:
    rows = [row(1), {"id": "no-messages"}, {"id": "x", "messages": messages()[:2]}]
    out, stats = build_split(FakeTokenizer(), rows, max_len=1000, split="val")
    assert len(out) == 1 and stats.dropped_bad_shape == 2 and stats.kept == 1


def test_stats_report_lengths_and_token_counts() -> None:
    rows = [row(i) for i in range(10)]
    out, stats = build_split(FakeTokenizer(), rows, max_len=1000, split="train")
    js = stats.to_json()
    assert js["kept"] == 10
    assert js["tokens_total"] == sum(len(t.input_ids) for t in out)
    assert js["completion_tokens_total"] == 10 * len("SELECT 1<|im_end|>")
    assert js["length_tokens"]["max"] >= js["length_tokens"]["p50"] > 0


def test_subset_is_deterministic_and_nested() -> None:
    rows = [row(i) for i in range(200)]
    a = select_subset(rows, 50, seed=0)
    assert a == select_subset(rows, 50, seed=0)
    assert a != select_subset(rows, 50, seed=1)
    bigger = {r["id"] for r in select_subset(rows, 100, seed=0)}
    assert {r["id"] for r in a} <= bigger  # smaller data-size ablation sits inside the larger one
    assert len(select_subset(rows, None, 0)) == 200
    assert len(select_subset(rows, 500, 0)) == 200


def test_subset_does_not_depend_on_input_order() -> None:
    rows = [row(i) for i in range(100)]
    forward = {r["id"] for r in select_subset(rows, 30, 0)}
    backward = {r["id"] for r in select_subset(list(reversed(rows)), 30, 0)}
    assert forward == backward


def test_load_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps(row(1)) + "\n\n" + json.dumps(row(2)) + "\n", encoding="utf8")
    assert [r["id"] for r in load_jsonl(f)] == ["r1", "r2"]
    with pytest.raises(DataError, match="not found"):
        load_jsonl(tmp_path / "missing.jsonl")
    f.write_text("{not json}\n", encoding="utf8")
    with pytest.raises(DataError, match=r"d\.jsonl:1"):
        load_jsonl(f)
