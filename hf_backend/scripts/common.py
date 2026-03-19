"""Shared helpers for data preparation and training scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from pypinyin import Style, lazy_pinyin


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_pinyin_syllable(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def chinese_phrases(text: str, min_chars: int = 2, max_chars: int = 32) -> List[str]:
    phrases: List[str] = []
    for match in CHINESE_RE.findall(text or ""):
        if min_chars <= len(match) <= max_chars:
            phrases.append(match)
    return phrases


def text_to_pinyin_syllables(text: str) -> List[str]:
    syllables: List[str] = []
    for char in text:
        if not ("\u4e00" <= char <= "\u9fff"):
            return []
        pieces = lazy_pinyin(char, style=Style.NORMAL, strict=False, errors="ignore")
        if not pieces:
            return []
        normalized = normalize_pinyin_syllable(pieces[0])
        if not normalized:
            return []
        syllables.append(normalized)
    return syllables
