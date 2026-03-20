"""Report telemetry-based acceptance and ranking metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.telemetry import JsonlTelemetryStore
from common import backend_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feedback telemetry metrics.")
    parser.add_argument(
        "--telemetry-dir",
        default=str((backend_root() / "telemetry").resolve()),
        help="Directory containing predictions.jsonl and feedback.jsonl.",
    )
    parser.add_argument("--max-events", type=int, default=5000, help="Limit to the most recent N events.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    telemetry_dir = Path(args.telemetry_dir).expanduser().resolve()
    store = JsonlTelemetryStore(enabled=True, root_dir=telemetry_dir)
    summary = store.summarize(max_events=args.max_events)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
