from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DATA_DIR,
    DEFAULT_MODEL_ID,
    QA_PREP_DIR,
    build_run_dir,
    build_sft_prompt,
    safe_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate valid/holdout prediction CSVs from a completed finetuning run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def build_doc_lookup() -> dict[str, dict[str, str]]:
    doc_df = load_csv(DATA_DIR / "apartment_chatbot_v3.csv")
    lookup: dict[str, dict[str, str]] = {}
    for _, row in doc_df.iterrows():
        doc_id = safe_text(row.get("문서ID"))
        if not doc_id or doc_id in lookup:
            continue
        lookup[doc_id] = {
            "아파트명": safe_text(row.get("아파트명")),
            "description": safe_text(row.get("description")),
            "가격요약": safe_text(row.get("가격요약")),
            "교통_비교요약": safe_text(row.get("교통_비교요약")),
            "정책특이사항_설명": safe_text(row.get("정책특이사항_설명")),
            "답변가능범위": safe_text(row.get("답변가능범위")),
            "데이터기준일": safe_text(row.get("데이터기준일")),
        }
    return lookup


def build_context_prompt(row: dict[str, Any], doc_lookup: dict[str, dict[str, str]], tokenizer: AutoTokenizer) -> str:
    question = safe_text(row.get("question"))
    cited_doc_ids = [safe_text(part) for part in safe_text(row.get("cited_doc_ids")).split("|") if safe_text(part)]
    doc_lines: list[str] = []
    answer_scope = safe_text(row.get("query_type"))
    match_status = safe_text(row.get("match_status"))
    must_not_recommend = safe_text(row.get("must_not_recommend"))
    must_disclose_limit = safe_text(row.get("must_disclose_limit"))
    used_fields = safe_text(row.get("used_fields"))
    top_doc_id = safe_text(row.get("top_doc_id"))

    for doc_id in cited_doc_ids[:3]:
        doc = doc_lookup.get(doc_id)
        if not doc:
            continue
        detail_parts = [
            f"문서ID={doc_id}",
            f"아파트명={doc.get('아파트명', '')}",
            f"description={doc.get('description', '')}",
        ]
        if doc.get("가격요약"):
            detail_parts.append(f"가격요약={doc['가격요약']}")
        if doc.get("교통_비교요약"):
            detail_parts.append(f"교통요약={doc['교통_비교요약']}")
        if doc.get("정책특이사항_설명"):
            detail_parts.append(f"정책요약={doc['정책특이사항_설명']}")
        if doc.get("답변가능범위"):
            detail_parts.append(f"답변가능범위={doc['답변가능범위']}")
        if doc.get("데이터기준일"):
            detail_parts.append(f"데이터기준일={doc['데이터기준일']}")
        doc_lines.append(" | ".join(detail_parts))

    instruction_lines = [
        "당신은 부동산 데이터셋 기반 챗봇의 post-train evaluator입니다.",
        "반드시 제공된 데이터 문맥과 row contract 안에서만 답변하세요.",
        "일반적인 부동산 조언, 외부 사이트 추천, 실시간 정보 부재 변명은 금지합니다.",
        f"query_type={answer_scope}",
        f"match_status={match_status}",
        f"used_fields={used_fields}",
        f"top_doc_id={top_doc_id}",
        f"must_not_recommend={must_not_recommend}",
        f"must_disclose_limit={must_disclose_limit}",
    ]
    if must_not_recommend == "Y":
        instruction_lines.append("추천을 하면 안 됩니다. 데이터 기준 한계와 답변 가능 범위를 반드시 명시하세요.")
    if must_disclose_limit == "Y":
        instruction_lines.append("답변 안에 반드시 '데이터 기준' 또는 '답변 가능 범위' 수준의 한계 고지를 포함하세요.")
    if answer_scope == "GENERAL_RETRIEVAL_QA":
        instruction_lines.append("질문에 맞는 문서 요약을 3~5문장으로 grounded하게 설명하세요.")
    else:
        instruction_lines.append("질문 조건에 맞는 후보나 제한 사유를 데이터 문맥 기준으로 짧고 명확하게 답하세요.")

    prompt_body = "\n".join(
        [
            "\n".join(instruction_lines),
            "",
            f"질문: {question}",
            "",
            "문서 문맥:",
            *([f"- {line}" for line in doc_lines] if doc_lines else ["- 문서 문맥 없음"]),
            "",
            "출력 규칙:",
            "- 한국어로 답변",
            "- 제공된 문서 문맥 밖의 사실을 만들지 않기",
            "- 가능하면 데이터 기준일 또는 답변 가능 범위를 자연스럽게 포함",
            "- 불필요한 서론 없이 바로 답변",
        ]
    )
    return build_sft_prompt(prompt_body, tokenizer=tokenizer)


def build_fallback_answer(row: dict[str, Any], doc_lookup: dict[str, dict[str, str]]) -> str:
    cited_doc_ids = [safe_text(part) for part in safe_text(row.get("cited_doc_ids")).split("|") if safe_text(part)]
    docs = [doc_lookup[doc_id] for doc_id in cited_doc_ids if doc_id in doc_lookup]
    must_disclose_limit = safe_text(row.get("must_disclose_limit")) == "Y"
    must_not_recommend = safe_text(row.get("must_not_recommend")) == "Y"
    query_type = safe_text(row.get("query_type"))

    if docs:
        primary = docs[0]
        description = safe_text(primary.get("description"))
        answer_scope = safe_text(primary.get("답변가능범위"))
        data_date = safe_text(primary.get("데이터기준일"))
        if query_type == "GENERAL_RETRIEVAL_QA":
            parts = [description] if description else []
            if data_date:
                parts.append(f"데이터 기준: {data_date}")
            if answer_scope:
                parts.append(f"답변 가능 범위: {answer_scope}")
            return "\n".join(part for part in parts if part)
        if must_not_recommend:
            parts = ["현재 데이터 기준으로는 추천을 확정할 수 없습니다."]
            if description:
                parts.append(description)
            if data_date:
                parts.append(f"데이터 기준: {data_date}")
            if answer_scope:
                parts.append(f"답변 가능 범위: {answer_scope}")
            return "\n".join(parts)
        if description:
            parts = [description]
            if data_date:
                parts.append(f"데이터 기준: {data_date}")
            if answer_scope:
                parts.append(f"답변 가능 범위: {answer_scope}")
            return "\n".join(parts)

    if must_disclose_limit or must_not_recommend:
        return "현재 데이터 기준으로는 답변을 확정할 수 없습니다.\n데이터 기준: 제공된 문서 문맥 없음"
    return "제공된 문서 문맥이 없어 답변을 생성하지 못했습니다."


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = build_context_prompt(row, doc_lookup=doc_lookup, tokenizer=tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)
    new_tokens = generated[0][encoded["input_ids"].shape[1] :]
    decoded = safe_text(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return decoded or build_fallback_answer(row, doc_lookup)


def write_predictions(
    source_df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: Path,
    doc_lookup: dict[str, dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> None:
    records = []
    for _, row in source_df.iterrows():
        record = row.to_dict()
        record["answer"] = generate_answer(
            model,
            tokenizer,
            row=record,
            doc_lookup=doc_lookup,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        records.append(record)
    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    final_dir = run_dir / "final"
    if not final_dir.exists():
        raise FileNotFoundError(f"Final finetuned model directory does not exist: {final_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(final_dir), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(final_dir),
        local_files_only=True,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    doc_lookup = build_doc_lookup()

    training_candidates = load_csv(QA_PREP_DIR / f"training_candidates_{args.model}.csv")
    valid_df = training_candidates[training_candidates["split"] == "valid"].copy()
    grounded_df = load_csv(QA_PREP_DIR / "holdout_grounded_generation.csv")
    safety_df = load_csv(QA_PREP_DIR / "holdout_edge_safety.csv")

    valid_out = run_dir / "valid_predictions.csv"
    grounded_out = run_dir / "grounded_holdout_predictions.csv"
    safety_out = run_dir / "edge_safety_holdout_predictions.csv"

    write_predictions(valid_df, model, tokenizer, valid_out, doc_lookup, args.max_new_tokens, args.temperature)
    write_predictions(grounded_df, model, tokenizer, grounded_out, doc_lookup, args.max_new_tokens, args.temperature)
    write_predictions(safety_df, model, tokenizer, safety_out, doc_lookup, args.max_new_tokens, args.temperature)

    print(f"Saved valid predictions to {valid_out}")
    print(f"Saved grounded holdout predictions to {grounded_out}")
    print(f"Saved edge safety predictions to {safety_out}")


if __name__ == "__main__":
    main()
