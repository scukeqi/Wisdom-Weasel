"""Download a public pinyin corpus seed set for SFT preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from common import backend_root, ensure_parent, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the seed pinyin dataset.")
    parser.add_argument("--dataset-id", default="Duyu/Pinyin-Hanzi", help="Hugging Face dataset ID.")
    parser.add_argument(
        "--output-path",
        default=str((backend_root() / "data" / "raw" / "duyu_pinyin_hanzi.jsonl").resolve()),
        help="Where to write the normalized JSONL file.",
    )
    parser.add_argument("--max-rows", type=int, default=250000, help="Maximum rows to export.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before sampling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    ensure_parent(output_path)

    dataset = load_dataset(args.dataset_id, split="train")
    column_names = list(dataset.column_names)
    if len(column_names) < 2:
        raise RuntimeError(f"Unexpected dataset schema for {args.dataset_id}: {column_names}")

    text_column, pinyin_column = column_names[:2]
    if args.max_rows > 0 and len(dataset) > args.max_rows:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_rows))

    def rows():
        for record in dataset:
            yield {
                "text": str(record[text_column]).strip(),
                "raw_pinyin": str(record[pinyin_column]).strip(),
                "source_dataset": args.dataset_id,
            }

    count = write_jsonl(output_path, rows())
    manifest = {
        "dataset_id": args.dataset_id,
        "output_path": str(output_path),
        "rows": count,
        "columns": {
            "text_column": text_column,
            "pinyin_column": pinyin_column,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
