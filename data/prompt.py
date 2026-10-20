"""The single prompt format used for SFT, DPO, baselines and serving.

Training and evaluation must build prompts with these functions, so the model is scored
on the same format it was trained on.
"""

from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema and a question, write one "
    "SQLite SELECT query that answers the question. Reply with the SQL only, no explanation."
)

PROMPT_VERSION = "v1"

Message = dict[str, str]


def user_prompt(schema: str, question: str, evidence: str = "") -> str:
    parts = [f"Database schema:\n{schema.strip()}"]
    if evidence.strip():
        parts.append(f"Hint: {evidence.strip()}")
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


def build_messages(schema: str, question: str, evidence: str = "",
                   answer_sql: str | None = None) -> list[Message]:
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(schema, question, evidence)},
    ]
    if answer_sql is not None:
        messages.append({"role": "assistant", "content": answer_sql})
    return messages


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(completion: str) -> str:
    """Pull the SQL out of a model reply: the first fenced block if any, else the full text.
    Used at inference time; training targets are bare SQL."""
    match = _FENCE.search(completion)
    sql = match.group(1) if match else completion
    return sql.strip().rstrip(";").strip()
