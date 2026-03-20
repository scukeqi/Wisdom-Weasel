"""
推理服务
"""
import torch
from typing import List, Dict

from transformers import LogitsProcessorList

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from pinyin_constraint import PinyinConstraintLogitsProcessor
    from model_manager import ModelManager
    from config import get_config
    from prompting import PromptBuilder
else:
    from .pinyin_constraint import PinyinConstraintLogitsProcessor
    from .model_manager import ModelManager
    from .config import get_config
    from .prompting import PromptBuilder
import os


class InferenceService:
    """LLM 推理服务"""

    def __init__(
        self,
        model_manager: ModelManager,
        use_chat_template: bool = True,
    ):
        self.model_manager = model_manager
        self.use_chat_template = use_chat_template
        self.config = get_config()
        self.prompt_builder = PromptBuilder(system_prompt=self.config.get_system_prompt())

    def _prepare_inputs(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        
        tokenizer = self.model_manager.get_tokenizer()
        device = self.model_manager.get_device()

        if self.use_chat_template and messages:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            text = messages[0]["content"] if messages else ""
            inputs = tokenizer(text, return_tensors="pt", return_dict=True)

        return {k: v.to(device) for k, v in inputs.items()}

    def _generate_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        pinyin_constraints: List[str] | None = None,
        max_new_tokens_override: int | None = None,
    ) -> List[str]:
        pinyin_constraints = pinyin_constraints or []
        model = self.model_manager.get_model()
        tokenizer = self.model_manager.get_tokenizer()
        input_len = inputs["input_ids"].shape[1]

        max_new_tokens = (
            max_new_tokens_override
            if max_new_tokens_override is not None
            else (
                self.config.get_max_new_tokens_no_constraint()
                if len(pinyin_constraints) == 0
                else len(pinyin_constraints)
            )
        )
        num_beams = max(1, int(self.config.get_num_beams()))
        num_return_sequences = max(1, int(self.config.get_num_return_sequences()))

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }

        if num_beams <= 1:
            gen_kwargs["num_beams"] = 1
            gen_kwargs["num_return_sequences"] = 1
        else:
            gen_kwargs.update({
                "num_beams": num_beams,
                "num_return_sequences": min(num_return_sequences, num_beams),
                "num_beam_groups": self.config.get_num_beam_groups(),
                "diversity_penalty": (
                    self.config.get_diversity_penalty_no_constraint()
                    if len(pinyin_constraints) == 0
                    else self.config.get_diversity_penalty_with_constraint()
                ),
                "custom_generate": os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "beam_search")
                ),
            })

        if len(pinyin_constraints) > 0:
            pinyin_mapping = self.model_manager.get_pinyins_mapping()
            pinyin_processor = PinyinConstraintLogitsProcessor(
                prompt_length=input_len,
                pinyin_constraints=pinyin_constraints,
                mapping=pinyin_mapping
            )
            gen_kwargs["logits_processor"] = LogitsProcessorList([pinyin_processor])

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                trust_remote_code=True,
                **gen_kwargs,
            )

        results: List[str] = []
        seen = set()
        for i in outputs:
            generated_tokens = i[input_len:]
            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                errors='ignore'
            ).replace('�', '').strip()
            if generated_text and generated_text not in seen:
                seen.add(generated_text)
                results.append(generated_text)

        return results


    def base_model_generate(
        self,
        prompt: str,
        pinyin_constraints: List[str] | None = None,
    ):
        """使用基础模型进行生成，支持拼音约束"""
        pinyin_constraints = pinyin_constraints or []
        tokenizer = self.model_manager.get_tokenizer()
        device = self.model_manager.get_device()

        if self.use_chat_template:
            messages = self.prompt_builder.build_messages(prompt, pinyin_constraints)
            inputs = self._prepare_inputs(messages, add_generation_prompt=True)
        else:
            rendered_prompt = self.prompt_builder.build_plaintext_prompt(prompt, pinyin_constraints)
            inputs = tokenizer(rendered_prompt, return_tensors="pt", return_dict=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return self._generate_from_inputs(inputs, pinyin_constraints=pinyin_constraints)

    def openai_compatible_generate(
        self,
        rendered_prompt: str,
        max_new_tokens: int | None = None,
    ) -> List[str]:
        tokenizer = self.model_manager.get_tokenizer()
        device = self.model_manager.get_device()
        inputs = tokenizer(rendered_prompt, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return self._generate_from_inputs(
            inputs,
            pinyin_constraints=[],
            max_new_tokens_override=max_new_tokens,
        )

if __name__ == "__main__":
    model_name = r"C:\Users\Ken\Downloads\Qwen3_4B"
    model_manager = ModelManager(model_name)
    model_manager.load_model()

    inferServer=InferenceService(model_manager)
    res=inferServer.base_model_generate("计算机视觉是", pinyin_constraints=['ji','su'])
    for r in res:
        print(r)
