"""QLoRA fine-tuning entrypoint for Qwen3.5-4B-Base."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.prompting import PromptBuilder
from common import backend_root, read_jsonl


DEFAULT_CONFIG_PATH = backend_root() / "training" / "train_config.sample.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5-4B-Base with QLoRA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH.resolve()), help="Training config JSON path.")
    parser.add_argument("--base-model-id", default=None, help="Override base model ID.")
    parser.add_argument("--dataset-path", default=None, help="Override dataset JSONL path.")
    parser.add_argument("--output-dir", default=None, help="Override adapter output directory.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Optional checkpoint directory.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Training config must be a JSON object.")
    return config


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        records.extend(read_jsonl(path))
    if not records:
        raise RuntimeError("No training records were loaded.")
    return records


def build_training_texts(
    record: Dict[str, object],
    tokenizer,
    prompt_builder: PromptBuilder,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    context = str(record.get("context", "")).strip()
    answer = str(record.get("answer", "")).strip()
    pinyin_constraints = [str(item).strip() for item in record.get("pinyin_constraints", []) if str(item).strip()]

    prompt_messages = prompt_builder.build_messages(context, pinyin_constraints)
    full_messages = prompt_builder.build_messages(context, pinyin_constraints, answer=answer)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    full_encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    prompt_encoding = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    labels = list(full_encoding["input_ids"])
    prompt_len = min(len(prompt_encoding["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    if all(label == -100 for label in labels):
        raise ValueError("Prompt consumed the full sequence; no supervised labels remain.")

    return {
        "input_ids": list(full_encoding["input_ids"]),
        "attention_mask": list(full_encoding["attention_mask"]),
        "labels": labels,
    }


@dataclass
class SupervisedDataCollator:
    tokenizer: object

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for row_idx, row_labels in enumerate(labels):
            padded_labels[row_idx, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
        batch["labels"] = padded_labels
        return batch


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    base_model_id = args.base_model_id or str(config.get("base_model_id") or "Qwen/Qwen3.5-4B-Base")
    dataset_paths = [Path(args.dataset_path).expanduser().resolve()] if args.dataset_path else []
    if not dataset_paths:
        dataset_paths.append(Path(str(config.get("dataset_path"))).expanduser().resolve())
    feedback_dataset_path = str(config.get("feedback_dataset_path") or "").strip()
    if feedback_dataset_path:
        dataset_paths.append(Path(feedback_dataset_path).expanduser().resolve())

    output_dir = Path(args.output_dir or str(config.get("output_dir"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq_length = int(config.get("max_seq_length", 256))
    eval_ratio = float(config.get("eval_ratio", 0.02))
    seed = int(config.get("seed", 42))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    prompt_builder = PromptBuilder(system_prompt=str(config.get("system_prompt") or PromptBuilder().system_prompt))

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = bool(config.get("load_in_4bit", True))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = config.get("lora", {}) or {}
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        target_modules=list(
            lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_records = load_records(dataset_paths)
    dataset = Dataset.from_list(raw_records)

    def tokenize_record(record):
        try:
            return build_training_texts(record, tokenizer, prompt_builder, max_seq_length=max_seq_length)
        except ValueError:
            return {"input_ids": [], "attention_mask": [], "labels": []}

    tokenized = dataset.map(tokenize_record, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 0)

    if len(tokenized) < 10:
        raise RuntimeError("Not enough usable training rows after tokenization.")

    if 0.0 < eval_ratio < 1.0 and len(tokenized) > 100:
        split = tokenized.train_test_split(test_size=eval_ratio, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        eval_strategy = "steps"
    else:
        train_dataset = tokenized
        eval_dataset = None
        eval_strategy = "no"

    train_args_config = dict(config.get("training_args", {}) or {})
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_args_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_args_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_args_config.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(train_args_config.get("num_train_epochs", 1.0)),
        learning_rate=float(train_args_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_args_config.get("warmup_ratio", 0.03)),
        weight_decay=float(train_args_config.get("weight_decay", 0.0)),
        logging_steps=int(train_args_config.get("logging_steps", 10)),
        save_steps=int(train_args_config.get("save_steps", 100)),
        save_total_limit=int(train_args_config.get("save_total_limit", 2)),
        eval_strategy=eval_strategy,
        eval_steps=int(train_args_config.get("eval_steps", 100)),
        bf16=bool(train_args_config.get("bf16", torch.cuda.is_available())),
        fp16=bool(train_args_config.get("fp16", False)),
        gradient_checkpointing=bool(train_args_config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs=train_args_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        report_to=train_args_config.get("report_to", "none"),
        lr_scheduler_type=str(train_args_config.get("lr_scheduler_type", "cosine")),
        optim=str(train_args_config.get("optim", "paged_adamw_8bit")),
        max_grad_norm=float(train_args_config.get("max_grad_norm", 0.3)),
        seed=seed,
        remove_unused_columns=False,
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=SupervisedDataCollator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "base_model_id": base_model_id,
        "output_dir": str(output_dir),
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "dataset_paths": [str(path) for path in dataset_paths if path.exists()],
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
