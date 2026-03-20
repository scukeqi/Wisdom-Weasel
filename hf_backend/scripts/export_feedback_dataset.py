"""Convert feedback telemetry logs into fine-tuning datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from common import backend_root, read_jsonl, text_to_pinyin_syllables, write_jsonl


DEFAULT_TELEMETRY_DIR = backend_root() / "telemetry"
DEFAULT_SFT_OUTPUT = backend_root() / "data" / "processed" / "feedback_sft.jsonl"
DEFAULT_PAIR_OUTPUT = backend_root() / "data" / "processed" / "feedback_pairs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feedback logs into SFT and preference datasets.")
    parser.add_argument(
        "--feedback-log",
        default=str((DEFAULT_TELEMETRY_DIR / "feedback.jsonl").resolve()),
        help="Path to feedback.jsonl.",
    )
    parser.add_argument("--sft-output", default=str(DEFAULT_SFT_OUTPUT.resolve()), help="SFT JSONL output path.")
    parser.add_argument("--pair-output", default=str(DEFAULT_PAIR_OUTPUT.resolve()), help="Preference-pair JSONL path.")
    return parser.parse_args()


def resolve_answer(record: Dict[str, object]) -> Optional[str]:
    expected = str(record.get("expected_text") or "").strip()
    if expected:
        return expected
    selected = str(record.get("selected_candidate") or "").strip()
    if selected:
        return selected
    return None


def build_sft_record(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    answer = resolve_answer(record)
    pinyin_constraints = [str(item).strip() for item in record.get("pinyin_constraints", []) if str(item).strip()]
    if not answer or not pinyin_constraints:
        return None
    if len(text_to_pinyin_syllables(answer)) != len(answer):
        return None

    return {
        "context": str(record.get("prompt", "")).strip(),
        "pinyin_constraints": pinyin_constraints,
        "answer": answer,
        "source": f"feedback:{record.get('source', 'user')}",
        "metadata": {
            "request_id": record.get("request_id"),
            "accepted": record.get("accepted"),
            "shown_candidates": record.get("shown_candidates", []),
        },
    }


def build_preference_records(record: Dict[str, object]) -> List[Dict[str, object]]:
    chosen = resolve_answer(record)
    if not chosen:
        return []

    prompt = str(record.get("prompt", "")).strip()
    pinyin_constraints = [str(item).strip() for item in record.get("pinyin_constraints", []) if str(item).strip()]
    rejected = [str(item).strip() for item in record.get("rejected_candidates", []) if str(item).strip()]
    shown = [str(item).strip() for item in record.get("shown_candidates", []) if str(item).strip()]

    if not rejected:
        rejected = [candidate for candidate in shown if candidate and candidate != chosen]

    pairs: List[Dict[str, object]] = []
    for rejected_candidate in rejected:
        if not rejected_candidate or rejected_candidate == chosen:
            continue
        pairs.append(
            {
                "context": prompt,
                "pinyin_constraints": pinyin_constraints,
                "chosen": chosen,
                "rejected": rejected_candidate,
                "source": f"feedback:{record.get('source', 'user')}",
                "metadata": {
                    "request_id": record.get("request_id"),
                    "accepted": record.get("accepted"),
                },
            }
        )
    return pairs


def main() -> int:
    args = parse_args()
    feedback_log = Path(args.feedback_log).expanduser().resolve()
    sft_output = Path(args.sft_output).expanduser().resolve()
    pair_output = Path(args.pair_output).expanduser().resolve()

    if not feedback_log.exists():
        raise FileNotFoundError(f"Feedback log not found: {feedback_log}")

    feedback_records = list(read_jsonl(feedback_log))
    sft_records = [record for raw in feedback_records if (record := build_sft_record(raw))]

    pair_records: List[Dict[str, object]] = []
    for raw in feedback_records:
        pair_records.extend(build_preference_records(raw))

    sft_count = write_jsonl(sft_output, sft_records)
    pair_count = write_jsonl(pair_output, pair_records)
    summary = {
        "feedback_log": str(feedback_log),
        "sft_output": str(sft_output),
        "pair_output": str(pair_output),
        "sft_records": sft_count,
        "pair_records": pair_count,
    }
    sft_output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
