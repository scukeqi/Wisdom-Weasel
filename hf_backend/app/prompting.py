"""Prompt builders shared by inference and fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


DEFAULT_SYSTEM_PROMPT = (
    "你是 Wisdom Weasel 输入法候选生成模型。"
    "请根据历史上下文和当前拼音，输出最自然、最贴合用户意图的中文候选。"
    "只输出候选文本本身，不要解释，不要编号，不要额外标点。"
)


@dataclass(slots=True)
class PromptBuilder:
    """Builds a stable prompt format for both training and inference."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def build_messages(
        self,
        context: str,
        pinyin_constraints: Sequence[str],
        answer: str | None = None,
    ) -> List[Dict[str, str]]:
        normalized_context = context.strip() or "（空上下文）"
        normalized_pinyin = " ".join(part.strip() for part in pinyin_constraints if part.strip())
        normalized_pinyin = normalized_pinyin or "（无拼音约束）"

        user_content = (
            "请根据输入法上下文预测当前拼音对应的中文候选。\n\n"
            f"【历史上下文】\n{normalized_context}\n\n"
            f"【当前拼音】\n{normalized_pinyin}\n\n"
            "【输出要求】\n"
            "1. 只输出中文候选文本。\n"
            "2. 输出必须与拼音严格对应。\n"
            "3. 不要添加解释、引号、序号或额外空格。"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer.strip()})
        return messages

    def build_plaintext_prompt(self, context: str, pinyin_constraints: Sequence[str]) -> str:
        normalized_context = context.strip() or "（空上下文）"
        normalized_pinyin = " ".join(part.strip() for part in pinyin_constraints if part.strip())
        normalized_pinyin = normalized_pinyin or "（无拼音约束）"
        return (
            f"{self.system_prompt}\n\n"
            "请根据输入法上下文预测当前拼音对应的中文候选。\n"
            f"历史上下文：{normalized_context}\n"
            f"当前拼音：{normalized_pinyin}\n"
            "候选："
        )
