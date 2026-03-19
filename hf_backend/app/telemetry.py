"""Lightweight JSONL telemetry store for prediction and feedback loops."""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlTelemetryStore:
    """Appends telemetry events to JSONL files and computes simple summaries."""

    def __init__(self, enabled: bool, root_dir: Path):
        self.enabled = enabled
        self.root_dir = root_dir
        self._lock = threading.Lock()

        self.prediction_path = self.root_dir / "predictions.jsonl"
        self.feedback_path = self.root_dir / "feedback.jsonl"

        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def is_enabled(self) -> bool:
        return self.enabled

    def append_prediction(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"event_type": "prediction", "timestamp": _utc_now(), **event}
        self._append_jsonl(self.prediction_path, payload)

    def append_feedback(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"event_type": "feedback", "timestamp": _utc_now(), **event}
        self._append_jsonl(self.feedback_path, payload)

    def summarize(self, max_events: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}

        predictions = self._read_jsonl(self.prediction_path, max_events=max_events)
        feedback = self._read_jsonl(self.feedback_path, max_events=max_events)

        accepted = [item for item in feedback if item.get("accepted") is True]
        rejected = [item for item in feedback if item.get("accepted") is False]

        labeled_events = 0
        top_hits = {1: 0, 3: 0, 5: 0}
        reciprocal_ranks: List[float] = []
        missed_targets = 0

        for item in feedback:
            target = (item.get("expected_text") or item.get("selected_candidate") or "").strip()
            if not target:
                continue
            labeled_events += 1
            shown = [str(candidate).strip() for candidate in item.get("shown_candidates", []) if str(candidate).strip()]
            try:
                rank = shown.index(target) + 1
            except ValueError:
                rank = None

            if rank is None:
                missed_targets += 1
                continue

            reciprocal_ranks.append(1.0 / rank)
            for k in top_hits:
                if rank <= k:
                    top_hits[k] += 1

        acceptance_rate = (len(accepted) / len(feedback)) if feedback else 0.0
        mean_reciprocal_rank = (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        )

        selected_counter = Counter(
            item.get("selected_candidate", "").strip()
            for item in accepted
            if item.get("selected_candidate")
        )

        return {
            "enabled": True,
            "prediction_events": len(predictions),
            "feedback_events": len(feedback),
            "accepted_feedback": len(accepted),
            "rejected_feedback": len(rejected),
            "acceptance_rate": round(acceptance_rate, 4),
            "labeled_feedback_events": labeled_events,
            "top1_hit_rate": round(self._safe_div(top_hits[1], labeled_events), 4),
            "top3_hit_rate": round(self._safe_div(top_hits[3], labeled_events), 4),
            "top5_hit_rate": round(self._safe_div(top_hits[5], labeled_events), 4),
            "mean_reciprocal_rank": round(mean_reciprocal_rank, 4),
            "targets_missed_in_candidates": missed_targets,
            "top_selected_candidates": selected_counter.most_common(10),
            "prediction_log_path": str(self.prediction_path),
            "feedback_log_path": str(self.feedback_path),
        }

    def _append_jsonl(self, path: Path, event: Dict[str, Any]) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if not denominator:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _read_jsonl(path: Path, max_events: Optional[int] = None) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if max_events is not None and max_events > 0:
            rows = rows[-max_events:]
        return rows
