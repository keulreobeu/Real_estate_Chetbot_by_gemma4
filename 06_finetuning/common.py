from __future__ import annotations

import hashlib
import json
import os
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
APARTMENT_DOCS_PATH = DATA_DIR / "apartment_chatbot_v3.csv"
CONTEXTUAL_SCHEMA_VERSION = "context-aware input schema v1"
CONTEXTUAL_VALIDATION_VERSION = "r4_disclosure_v1"


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


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def replace_file_atomically(source: Path, destination: Path) -> None:
    ensure_parent(destination)
    os.replace(source, destination)


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


def get_context_schema_defaults() -> dict[str, int | str]:
    return {
        "schema_version": CONTEXTUAL_SCHEMA_VERSION,
        "validation_version": CONTEXTUAL_VALIDATION_VERSION,
        "max_docs": 1,
        "max_description_chars": 12,
        "max_seq_length": 512,
    }


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


def build_sft_user_content(instruction: str, input_text: str = "") -> str:
    instruction_text = safe_text(instruction)
    input_value = safe_text(input_text)
    return instruction_text if not input_value else f"{instruction_text}\n\n{input_value}"


def compute_prompt_token_count(
    instruction: str,
    input_text: str,
    *,
    tokenizer: Any,
) -> int:
    user_content = build_sft_user_content(instruction, input_text)
    prompt = build_sft_prompt(user_content, tokenizer=tokenizer)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


def compute_full_training_token_count(
    instruction: str,
    input_text: str,
    output_text: str,
    *,
    tokenizer: Any,
) -> int:
    user_content = build_sft_user_content(instruction, input_text)
    full_text = build_sft_training_text(user_content, output_text, tokenizer=tokenizer)
    full_sequence_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    return full_sequence_tokens + (1 if getattr(tokenizer, "eos_token_id", None) is not None else 0)


def parse_pipe_values(value: Any, limit: int | None = None) -> list[str]:
    items = [safe_text(part) for part in safe_text(value).split("|") if safe_text(part)]
    return items[:limit] if limit is not None else items


def normalize_context_doc_ids(row: dict[str, Any], *, max_docs: int) -> list[str]:
    cited_doc_ids = parse_pipe_values(row.get("cited_doc_ids"))
    top_doc_id = safe_text(row.get("top_doc_id"))
    if top_doc_id and top_doc_id not in cited_doc_ids:
        cited_doc_ids = [top_doc_id, *cited_doc_ids]
    return cited_doc_ids[:max_docs]


def resolve_context_docs(
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_docs: int,
) -> list[tuple[str, dict[str, str]]]:
    resolved: list[tuple[str, dict[str, str]]] = []
    for doc_id in normalize_context_doc_ids(row, max_docs=max_docs):
        doc = doc_lookup.get(doc_id)
        if doc:
            resolved.append((doc_id, doc))
    return resolved


def clip_text(text: Any, max_chars: int) -> str:
    value = safe_text(text)
    if len(value) <= max_chars:
        return value
    clipped = value[: max(0, max_chars - 1)].rstrip()
    return f"{clipped}…"


def load_apartment_doc_lookup(path: Path | None = None) -> dict[str, dict[str, str]]:
    docs_path = path or APARTMENT_DOCS_PATH
    df = load_csv(docs_path)
    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        doc_id = safe_text(row.get("문서ID"))
        if not doc_id or doc_id in lookup:
            continue
        lookup[doc_id] = {
            "doc_id": doc_id,
            "apartment_name": safe_text(row.get("아파트명")),
            "description": safe_text(row.get("description")),
            "price_summary": safe_text(row.get("가격요약")),
            "traffic_summary": safe_text(row.get("교통_비교요약")),
            "policy_summary": safe_text(row.get("정책특이사항_설명")),
            "answer_scope": safe_text(row.get("답변가능범위")),
            "data_date": safe_text(row.get("데이터기준일")),
        }
    return lookup


def build_contextual_instruction(row: dict[str, Any]) -> str:
    must_not_recommend = safe_text(row.get("must_not_recommend")) == "Y"
    must_disclose_limit = safe_text(row.get("must_disclose_limit")) == "Y"
    parts = ["문맥만 사용. 추측 금지."]
    if must_not_recommend:
        parts.append("추천 금지.")
    if must_disclose_limit:
        parts.append("끝에 '데이터 기준:'과 '답변 가능 범위:'를 그대로 쓰기.")
    return " ".join(parts)


def build_contextual_output_rules(row: dict[str, Any]) -> list[str]:
    query_type = safe_text(row.get("query_type"))
    must_not_recommend = safe_text(row.get("must_not_recommend")) == "Y"
    must_disclose_limit = safe_text(row.get("must_disclose_limit")) == "Y"
    rules = [
        "- 한국어로 답변",
        "- 제공된 문서 문맥 밖의 사실을 만들지 않기",
        "- 일반적인 부동산 조언이나 외부 검색 유도 금지",
    ]
    if query_type == "GENERAL_RETRIEVAL_QA":
        rules.append("- 문서 설명을 3~5문장 grounded summary로 정리")
    else:
        rules.append("- 질문 조건에 맞는 후보 또는 제한 사유부터 바로 제시")
    if must_not_recommend:
        rules.append("- 추천 표현 금지, 데이터 한계 중심으로 설명")
    if must_disclose_limit:
        rules.append("- 반드시 '데이터 기준' 또는 '답변 가능 범위' 수준의 한계 고지 포함")
    return rules


def build_contextual_input(
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_docs: int = 2,
    max_description_chars: int = 180,
) -> str:
    top_doc_id = safe_text(row.get("top_doc_id"))
    cited_doc_ids = normalize_context_doc_ids(row, max_docs=max_docs)
    used_fields = safe_text(row.get("used_fields"))
    include_price = "price" in used_fields.lower() or "가격" in used_fields
    include_traffic = "distance" in used_fields.lower() or "교통" in used_fields or "거리" in used_fields
    include_policy = "정책" in used_fields or "policy" in used_fields.lower()

    compact_cited_doc_ids = cited_doc_ids
    context_lines: list[str] = []
    answer_scope = ""
    data_date = ""
    for _, (doc_id, doc) in enumerate(resolve_context_docs(row, doc_lookup, max_docs=max_docs), start=1):
        answer_scope = answer_scope or doc.get("answer_scope", "")
        data_date = data_date or doc.get("data_date", "")
        line_parts = [
            safe_text(doc.get("apartment_name", "")),
            clip_text(doc.get("description", ""), max_description_chars),
        ]
        extra_summary = ""
        if include_price and doc.get("price_summary"):
            extra_summary = clip_text(doc["price_summary"], 10)
        elif include_traffic and doc.get("traffic_summary"):
            extra_summary = clip_text(doc["traffic_summary"], 10)
        elif include_policy and doc.get("policy_summary"):
            extra_summary = clip_text(doc["policy_summary"], 10)
        if extra_summary:
            line_parts.append(extra_summary)
        context_lines.append("|".join([part for part in line_parts if part]))

    body_lines = [
        (
            f"q={safe_text(row.get('query_type'))};"
            f"m={safe_text(row.get('match_status'))};"
            f"n={safe_text(row.get('must_not_recommend'))};"
            f"d={safe_text(row.get('must_disclose_limit'))};"
            f"u={clip_text(used_fields, 12)}"
        ),
        f"t={top_doc_id};c={'|'.join(compact_cited_doc_ids)}",
        safe_text(row.get("question")),
    ]
    body_lines.extend(context_lines or ["ctx=none"])
    if safe_text(row.get("must_disclose_limit")) == "Y":
        body_lines.append("rule=끝에 데이터 기준: / 답변 가능 범위: 포함")
        if data_date:
            body_lines.append(f"dt={data_date}")
        if answer_scope:
            body_lines.append(f"sc={clip_text(answer_scope, 10)}")
    return "\n".join(body_lines)


def build_contextual_schema_markdown() -> str:
    defaults = get_context_schema_defaults()
    return f"""# Context-Aware Input Schema v1

## Purpose

- align stage 06 training rows with the row-aware post-train prediction contract
- keep the schema compressed enough for the local 8GB GPU path

## Shape

```text
q=...;m=...;n=...;d=...;u=...
t=...;c=...
question text
apartment_name|description|optional_summary
doc_2_context
dt=...
sc=...
rule=...
```

## Field rules

- always include the row contract fields in fixed compact order
- compact key map: `q=query_type`, `m=match_status`, `n=must_not_recommend`, `d=must_disclose_limit`, `u=used_fields`, `t=top_doc_id`, `c=cited_doc_ids`, `dt=data_date`, `sc=answer_scope`
- include at most the selected cited docs, defaulting to 1 for r3
- clip each doc description to a very short grounded summary
- include at most one compact extra summary per doc when `used_fields` suggests it matters
- only include disclosure rule and `dt=` / `sc=` when disclosure-limited rows need them
- rows with no resolvable cited document are excluded from the contextual train/valid build
- keep train and valid on the exact same schema builder

## Budget

- target prompt budget: informational only, full training sequence must fit within `max_seq_length`
- default docs: {defaults["max_docs"]}
- default max description chars: {defaults["max_description_chars"]}
- default max sequence length: {defaults["max_seq_length"]}
- no raw apartment row dumps
- no long keyword fields
- compact field keys are intentional to keep the full training sequence inside the 8GB-safe budget
"""


def contextual_summary_is_accepted(summary: dict[str, Any] | None) -> bool:
    if not isinstance(summary, dict):
        return False
    budget = summary.get("selected_schema_budget", {})
    return (
        summary.get("builder_pass") is True
        and summary.get("schema_status") == "accepted"
        and summary.get("validation_version") == CONTEXTUAL_VALIDATION_VERSION
        and isinstance(budget, dict)
        and all(key in budget for key in ("max_docs", "max_description_chars", "max_seq_length"))
        and "split_counts" in summary
    )
