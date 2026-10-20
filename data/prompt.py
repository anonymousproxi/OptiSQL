"""The single prompt format used for SFT, DPO, baselines and serving.

Training and evaluation must build prompts with build_prompt(), so the model is scored on
exactly the format it was trained on. Changing anything here (text, sample rows, value
cut-off) means bumping PROMPT_VERSION and regenerating the data.
"""

from __future__ import annotations

import re
from pathlib import Path

from data.schema import serialize_schema

PROMPT_VERSION = "v1"
SCHEMA_SAMPLE_ROWS = 3  # example rows per table in the schema text, fixed for this version

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema and a question, write one "
    "SQLite SELECT query that answers the question. Reply with the SQL only, no explanation."
)

Message = dict[str, str]

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", text).strip()


def user_prompt(schema: str, question: str, evidence: str = "") -> str:
    parts = [f"Database schema:\n{schema.strip()}"]
    if clean_text(evidence):
        parts.append(f"Hint: {clean_text(evidence)}")
    parts.append(f"Question: {clean_text(question)}")
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


def build_prompt(db_path: str | Path, question: str, evidence: str = "",
                 answer_sql: str | None = None) -> list[Message]:
    """Chat messages for one example. Use this everywhere (training, baselines, serving)."""
    schema = serialize_schema(Path(db_path), SCHEMA_SAMPLE_ROWS)
    return build_messages(schema, question, evidence, answer_sql)


_FENCE = re.compile(r"```[ \t]*(?:sqlite|sql)?[ \t]*\n?(.*?)(?:```|$)",
                    re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE | re.MULTILINE)


def extract_sql(completion: str) -> str:
    """Pull the SQL out of a model reply.

    Takes the first fenced block (closed or cut off), otherwise the text from the first line
    that starts with SELECT or WITH, otherwise the whole reply. Training targets are bare SQL,
    so a well-trained model hits the last case.
    """
    match = _FENCE.search(completion)
    if match:
        sql = match.group(1)
    else:
        start = _SQL_START.search(completion)
        sql = completion[start.start():] if start else completion
    return sql.strip().rstrip(";").strip()
