import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.telemetry import JsonlTelemetryStore


class TelemetryStoreTests(unittest.TestCase):
    def test_summary_computes_acceptance_and_hits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonlTelemetryStore(enabled=True, root_dir=Path(tmp_dir))
            store.append_prediction(
                {
                    "request_id": "abc",
                    "prompt": "上下文",
                    "pinyin_constraints": ["pin", "yin"],
                    "candidates": ["拼音", "频音"],
                    "responses": "拼音 频音",
                    "latency_ms": 10.0,
                    "metadata": {},
                }
            )
            store.append_feedback(
                {
                    "request_id": "abc",
                    "prompt": "上下文",
                    "pinyin_constraints": ["pin", "yin"],
                    "shown_candidates": ["拼音", "频音"],
                    "selected_candidate": "拼音",
                    "accepted": True,
                    "expected_text": None,
                    "rejected_candidates": ["频音"],
                    "source": "user",
                    "metadata": {},
                }
            )
            summary = store.summarize()
            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["prediction_events"], 1)
            self.assertEqual(summary["feedback_events"], 1)
            self.assertEqual(summary["acceptance_rate"], 1.0)
            self.assertEqual(summary["top1_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
