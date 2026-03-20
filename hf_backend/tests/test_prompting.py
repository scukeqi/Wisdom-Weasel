import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.prompting import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_build_messages_includes_context_and_pinyin(self):
        builder = PromptBuilder()
        messages = builder.build_messages("最近在调候选排序", ["pin", "yin"], answer="拼音")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("最近在调候选排序", messages[1]["content"])
        self.assertIn("pin yin", messages[1]["content"])
        self.assertEqual(messages[-1]["content"], "拼音")


if __name__ == "__main__":
    unittest.main()
