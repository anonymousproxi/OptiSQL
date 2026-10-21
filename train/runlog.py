"""Plain-JSON run records: what was run, on what, and what came out.

Tracking choice (PROGRESS.md): JSON files, not Weights & Biases. They need no account, can be
committed as evidence, and are what docs/RESULTS.md numbers are copied from. W&B can be added with
`report_to = "wandb"`; the JSON files are written either way.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate",
                    "bitsandbytes", "unsloth", "unsloth_zoo", "triton")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    dirty = bool(status) if status is not None else None
    return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}


def gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "note": "torch not installed"}
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {"cuda": True, "name": props.name,
            "total_memory_gb": round(props.total_memory / 2**30, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported())}


def environment(repo: Path) -> dict[str, Any]:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "gpu": gpu_info(), "git": git_state(repo)}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as fh:
        fh.write(json.dumps({"time": round(time.time(), 3), **record}, default=str) + "\n")
