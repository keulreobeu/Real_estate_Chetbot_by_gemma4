from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from common import (
    contextual_summary_is_accepted,
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    build_run_dir,
    build_sft_prompt,
    build_sft_training_text,
    build_sft_user_content,
    load_json,
    load_model_config,
)


class JsonlSftDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_seq_length: int) -> None:
        self.records: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        user_content = build_sft_user_content(record.get("instruction", ""), record.get("input", ""))
        prompt = build_sft_prompt(user_content, tokenizer=self.tokenizer)
        full_text = build_sft_training_text(
            user_content,
            str(record.get("output", "")),
            tokenizer=self.tokenizer,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        eos_id = self.tokenizer.eos_token_id
        input_ids = (full_ids + ([eos_id] if eos_id is not None else []))[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + full_ids[len(prompt_ids) :] + ([eos_id] if eos_id is not None else []))[
            : self.max_seq_length
        ]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class DataCollatorForCausalLm:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            batch_labels.append(item["labels"] + [-100] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stage 06 finetuning job from a frozen or run-local stage 06 manifest.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--training-scope",
        choices=["full", "gates_and_norms"],
        default="gates_and_norms",
        help="Parameter update scope. gates_and_norms is the memory-safe baseline for the local 8GB GPU.",
    )
    return parser.parse_args()


def configure_trainable_parameters(model: Any, training_scope: str) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_names: list[str] = []
    if training_scope == "full":
        for name, parameter in model.named_parameters():
            if name.startswith("model.language_model."):
                parameter.requires_grad = True
                trainable_names.append(name)
    elif training_scope == "gates_and_norms":
        for name, parameter in model.named_parameters():
            if not name.startswith("model.language_model."):
                continue
            if (
                ".per_layer_input_gate.weight" in name
                or ".per_layer_projection.weight" in name
                or name.endswith("input_layernorm.weight")
                or name.endswith("post_attention_layernorm.weight")
                or name.endswith("pre_feedforward_layernorm.weight")
                or name.endswith("post_feedforward_layernorm.weight")
                or name.endswith("post_per_layer_input_norm.weight")
                or name == "model.language_model.norm.weight"
            ):
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad = True
                trainable_names.append(name)
    else:
        raise ValueError(f"Unsupported training scope: {training_scope}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "training_scope": training_scope,
        "trainable_parameter_count": trainable_params,
        "total_parameter_count": total_params,
        "trainable_parameter_ratio": (trainable_params / total_params) if total_params else 0.0,
        "trainable_parameter_names_preview": trainable_names[:25],
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    manifest = load_json(run_dir / "manifest.json")
    config_path = run_dir / "config.json"
    train_log_path = run_dir / "train.log"
    checkpoints_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(train_log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("train_finetuning_baseline")

    model_config = load_model_config(args.model)
    model_source = model_config.get("local_dir") or model_config.get("hf_model_id")
    if not model_source:
        raise RuntimeError(f"Unable to resolve model source for {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_local_inputs = manifest.get("run_local_inputs", {})
    train_path = Path(run_local_inputs.get("selected_train_file", {}).get("path", manifest["frozen_inputs"]["train_file"]["path"]))
    valid_path = Path(run_local_inputs.get("selected_valid_file", {}).get("path", manifest["frozen_inputs"]["valid_file"]["path"]))
    train_path = train_path if train_path.is_absolute() else PROJECT_ROOT / train_path
    valid_path = valid_path if valid_path.is_absolute() else PROJECT_ROOT / valid_path
    contextual_selection_mode = run_local_inputs.get("contextual_selection_mode", "")
    if contextual_selection_mode == "accepted_contextual":
        contextual_summary_path = run_dir / "context_build_summary.json"
        contextual_summary = load_json(contextual_summary_path)
        if not contextual_summary_is_accepted(contextual_summary):
            raise RuntimeError("Manifest selected contextual inputs without an accepted context build summary.")
        expected_max_seq_length = int(contextual_summary["selected_schema_budget"]["max_seq_length"])
        if args.max_seq_length != expected_max_seq_length:
            raise RuntimeError(
                f"max_seq_length mismatch: trainer={args.max_seq_length} contextual_build={expected_max_seq_length}"
            )
    train_dataset = JsonlSftDataset(train_path, tokenizer, args.max_seq_length)
    eval_dataset = JsonlSftDataset(valid_path, tokenizer, args.max_seq_length)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=True,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    trainable_summary = configure_trainable_parameters(model, args.training_scope)

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    config_payload = {
        "run_id": args.run_id,
        "base_model_id": manifest.get("base_model_id", args.model),
        "training_method": "trainer_partial_finetune" if args.training_scope != "full" else "full_finetune_trainer",
        "adapter_or_full_finetune": "partial" if args.training_scope != "full" else "full",
        "train_file": str(train_path.relative_to(PROJECT_ROOT)) if train_path.is_absolute() else str(train_path),
        "valid_file": str(valid_path.relative_to(PROJECT_ROOT)) if valid_path.is_absolute() else str(valid_path),
        "grounded_holdout_file": manifest["frozen_inputs"]["grounded_holdout_file"]["path"],
        "edge_safety_holdout_file": manifest["frozen_inputs"]["edge_safety_holdout_file"]["path"],
        "trainable_parameter_summary": trainable_summary,
        "hyperparameters": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_length": args.max_seq_length,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "save_total_limit": args.save_total_limit,
            "training_scope": args.training_scope,
        },
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLm(tokenizer),
    )

    logger.info(
        "Starting baseline finetuning run %s with scope=%s trainable_params=%s ratio=%.6f",
        args.run_id,
        trainable_summary["training_scope"],
        trainable_summary["trainable_parameter_count"],
        trainable_summary["trainable_parameter_ratio"],
    )
    trainer.train()
    logger.info("Training completed, saving final artifact")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = trainer.evaluate()
    (run_dir / "training_result.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved final artifact to %s", final_dir)
    logger.info("Saved training_result.json")


if __name__ == "__main__":
    main()
