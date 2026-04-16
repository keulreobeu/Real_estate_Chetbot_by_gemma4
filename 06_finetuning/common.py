from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
QA_DIR = DATA_DIR / "qa"
QA_PREP_DIR = QA_DIR / "finetuning_prep"
FINETUNING_RUNS_DIR = QA_DIR / "finetuning_runs"
EVAL_DIR = DATA_DIR / "eval"
REPORT_DIR = PROJECT_ROOT / "00_Report"
DEFAULT_MODEL_ID = "gemma4_2b"
MODEL_CONFIG_PATH = PROJECT_ROOT / "02_gemma4_generation" / "config" / "models.local.json"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    return " ".join(safe_text(text).lower().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}


def build_run_dir(run_id: str) -> Path:
    return FINETUNING_RUNS_DIR / run_id


def get_default_frozen_paths(model_id: str) -> dict[str, Path]:
    return {
        "train_file": QA_PREP_DIR / f"train_{model_id}.jsonl",
        "valid_file": QA_PREP_DIR / f"valid_{model_id}.jsonl",
        "training_candidates_file": QA_PREP_DIR / f"training_candidates_{model_id}.csv",
        "dataset_summary_file": QA_PREP_DIR / f"dataset_summary_{model_id}.json",
        "readiness_file": QA_PREP_DIR / f"stage06_readiness_{model_id}.json",
        "grounded_holdout_file": QA_PREP_DIR / "holdout_grounded_generation.csv",
        "edge_safety_holdout_file": QA_PREP_DIR / "holdout_edge_safety.csv",
        "baseline_eval_predictions_file": EVAL_DIR / f"gemma4_generation_eval_predictions_{model_id}.csv",
        "baseline_edge_predictions_file": EVAL_DIR / f"gemma4_generation_edge_predictions_{model_id}.csv",
        "baseline_eval_metrics_file": EVAL_DIR / f"gemma4_generation_eval_metrics_{model_id}.json",
        "baseline_edge_metrics_file": EVAL_DIR / f"gemma4_generation_edge_metrics_{model_id}.json",
        "baseline_generation_summary_file": EVAL_DIR / "generation_optimization" / f"generation_optimization_summary_{model_id}.json",
    }


def describe_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def load_model_config(model_id: str) -> dict[str, Any]:
    payload = load_json(MODEL_CONFIG_PATH)
    models = payload.get("models", {})
    if model_id not in models:
        raise KeyError(f"Model config not found for {model_id}: {MODEL_CONFIG_PATH}")
    return dict(models[model_id])


def build_sft_prompt(question: str, tokenizer: Any | None = None) -> str:
    question_text = safe_text(question)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"질문:\n{question_text}\n\n답변:\n"


def build_sft_training_text(question: str, answer: str, tokenizer: Any | None = None) -> str:
    question_text = safe_text(question)
    answer_text = safe_text(answer)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": answer_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"질문:\n{question_text}\n\n답변:\n{answer_text}"
