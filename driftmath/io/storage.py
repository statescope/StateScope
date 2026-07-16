"""JSONL read/write helpers for pydantic records (and plain dicts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _to_line(record: BaseModel | dict) -> str:
    if isinstance(record, BaseModel):
        return record.model_dump_json()
    return json.dumps(record, ensure_ascii=False)


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> int:
    """Write ``records`` to a JSONL file (overwriting). Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(_to_line(r))
            f.write("\n")
            n += 1
    return n


def append_jsonl(path: str | Path, record: BaseModel | dict) -> None:
    """Append a single record to a JSONL file (creating it if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_to_line(record))
        f.write("\n")


def iter_jsonl(path: str | Path, model: Type[T] | None = None) -> Iterator:
    """Yield records from a JSONL file, parsing into ``model`` if given."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate_json(line) if model is not None else json.loads(line)


def read_jsonl(path: str | Path, model: Type[T] | None = None) -> list:
    """Read an entire JSONL file into a list (parsing into ``model`` if given)."""
    return list(iter_jsonl(path, model))
