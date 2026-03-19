"""Prepare SFT data for the Wisdom Weasel pinyin constrained use case."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import backend_root, chinese_phrases, project_root, read_jsonl, text_to_pinyin_syllables, write_jsonl


CURATED_PROJECT_PHRASES = backend_root() / "data" / "project_seed_phrases.txt"
DEFAULT_RAW_PATH = backend_root() / "data" / "raw" / "duyu_pinyin_hanzi.jsonl"
DEFAULT_OUTPUT_PATH = backend_root() / "data" / "processed" / "pinyin_sft.jsonl"
DEFAULT_FEEDBACK_SFT_PATH = backend_root() / "data" / "processed" / "feedback_sft.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SFT dataset used for QLoRA fine-tuning.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH.resolve()), help="Normalized raw JSONL path.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH.resolve()), help="Output SFT JSONL path.")
    parser.add_argument(
        "--feedback-path",
        default=str(DEFAULT_FEEDBACK_SFT_PATH.resolve()),
        help="Optional feedback SFT JSONL path to merge in if present.",
    )
    parser.add_argument("--max-examples", type=int, default=300000, help="Max seed examples from public corpus.")
    parser.add_argument("--max-target-chars", type=int, default=4, help="Maximum predicted Chinese characters.")
    parser.add_argument("--min-target-chars", type=int, default=1, help="Minimum predicted Chinese characters.")
    parser.add_argument("--repo-phrase-limit", type=int, default=200, help="Extra phrases mined from repo docs.")
    return parser.parse_args()


def build_example(
    context: str,
    target: str,
    source: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Optional[Dict[str, object]]:
    target = target.strip()
    if not target:
        return None

    pinyin_constraints = text_to_pinyin_syllables(target)
    if not pinyin_constraints or len(pinyin_constraints) != len(target):
        return None

    return {
        "context": context.strip(),
        "pinyin_constraints": pinyin_constraints,
        "answer": target,
        "source": source,
        "metadata": metadata or {},
    }


def build_seed_examples_from_text(
    text: str,
    max_target_chars: int,
    min_target_chars: int,
    source: str,
    metadata: Dict[str, object],
) -> Iterable[Dict[str, object]]:
    for phrase in chinese_phrases(text, min_chars=max(min_target_chars, 2), max_chars=24):
        limit = min(max_target_chars, len(phrase))
        for target_chars in range(min_target_chars, limit + 1):
            target = phrase[-target_chars:]
            context = phrase[:-target_chars]
            example = build_example(context=context, target=target, source=source, metadata=metadata)
            if example:
                yield example


def load_curated_project_phrases(path: Path) -> List[str]:
    if not path.exists():
        return []
    phrases: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase = line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        phrases.append(phrase)
    return phrases


def mine_repo_phrases(limit: int) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    files = [
        project_root() / "README.md",
        backend_root() / "README.md",
        project_root() / "CHANGELOG.md",
    ]

    for path in files:
        if not path.exists():
            continue
        seen_local = set()
        for phrase in chinese_phrases(path.read_text(encoding="utf-8"), min_chars=2, max_chars=12):
            if phrase in seen_local:
                continue
            seen_local.add(phrase)
            candidates.append((phrase, str(path)))
            if len(candidates) >= limit:
                return candidates
    return candidates


def merge_feedback_examples(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    feedback_path = Path(args.feedback_path).expanduser().resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw seed file not found: {raw_path}")

    examples: List[Dict[str, object]] = []
    seen = set()
    source_counter: Counter[str] = Counter()

    for record in read_jsonl(raw_path):
        for example in build_seed_examples_from_text(
            text=str(record.get("text", "")),
            max_target_chars=args.max_target_chars,
            min_target_chars=args.min_target_chars,
            source="duyu_pinyin_hanzi",
            metadata={"source_dataset": record.get("source_dataset")},
        ):
            key = (
                str(example["context"]),
                tuple(example["pinyin_constraints"]),
                str(example["answer"]),
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
            source_counter[str(example["source"])] += 1
            if args.max_examples > 0 and source_counter["duyu_pinyin_hanzi"] >= args.max_examples:
                break
        if args.max_examples > 0 and source_counter["duyu_pinyin_hanzi"] >= args.max_examples:
            break

    for phrase in load_curated_project_phrases(CURATED_PROJECT_PHRASES):
        example = build_example(
            context="",
            target=phrase,
            source="project_curated_phrase",
            metadata={"origin": str(CURATED_PROJECT_PHRASES)},
        )
        if not example:
            continue
        key = (str(example["context"]), tuple(example["pinyin_constraints"]), str(example["answer"]))
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        source_counter[str(example["source"])] += 1

    for phrase, origin in mine_repo_phrases(limit=args.repo_phrase_limit):
        example = build_example(
            context="",
            target=phrase,
            source="project_mined_phrase",
            metadata={"origin": origin},
        )
        if not example:
            continue
        key = (str(example["context"]), tuple(example["pinyin_constraints"]), str(example["answer"]))
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        source_counter[str(example["source"])] += 1

    for feedback_example in merge_feedback_examples(feedback_path):
        key = (
            str(feedback_example.get("context", "")),
            tuple(feedback_example.get("pinyin_constraints", [])),
            str(feedback_example.get("answer", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(feedback_example)
        source_counter[str(feedback_example.get("source", "feedback"))] += 1

    count = write_jsonl(output_path, examples)
    manifest = {
        "output_path": str(output_path),
        "total_examples": count,
        "source_breakdown": dict(source_counter),
        "raw_seed_path": str(raw_path),
        "feedback_path": str(feedback_path) if feedback_path.exists() else None,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
