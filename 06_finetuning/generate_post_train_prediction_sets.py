from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DEFAULT_MODEL_ID,
    QA_PREP_DIR,
    build_contextual_input,
    build_contextual_instruction,
    build_run_dir,
    build_sft_prompt,
    build_sft_user_content,
    contextual_summary_is_accepted,
    get_context_schema_defaults,
    load_apartment_doc_lookup,
    load_csv,
    load_json,
    normalize_context_doc_ids,
    resolve_context_docs,
    safe_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate valid/holdout prediction CSVs from a completed finetuning run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-description-chars", type=int, default=None)
    parser.add_argument("--stop-signal-path", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def resolve_prediction_budget(run_dir: Path, args: argparse.Namespace) -> tuple[int, int]:
    defaults = get_context_schema_defaults()
    summary_path = run_dir / "context_build_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    accepted_budget = summary.get("selected_schema_budget", {}) if contextual_summary_is_accepted(summary) else {}
    max_docs = int(args.max_docs if args.max_docs is not None else accepted_budget.get("max_docs", defaults["max_docs"]))
    max_description_chars = int(
        args.max_description_chars
        if args.max_description_chars is not None
        else accepted_budget.get("max_description_chars", defaults["max_description_chars"])
    )
    return max_docs, max_description_chars


def build_context_prompt(
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    tokenizer: AutoTokenizer,
    *,
    max_docs: int,
    max_description_chars: int,
) -> str:
    instruction = build_contextual_instruction(row)
    input_text = build_contextual_input(
        row,
        doc_lookup,
        max_docs=max_docs,
        max_description_chars=max_description_chars,
    )
    user_content = build_sft_user_content(instruction, input_text)
    return build_sft_prompt(user_content, tokenizer=tokenizer)


def build_fallback_answer(
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_docs: int,
) -> str:
    docs = [doc for _, doc in resolve_context_docs(row, doc_lookup, max_docs=max_docs)]
    must_disclose_limit = safe_text(row.get("must_disclose_limit")) == "Y"
    must_not_recommend = safe_text(row.get("must_not_recommend")) == "Y"
    query_type = safe_text(row.get("query_type"))

    if docs:
        primary = docs[0]
        description = safe_text(primary.get("description"))
        answer_scope = safe_text(primary.get("answer_scope"))
        data_date = safe_text(primary.get("data_date"))
        if query_type == "GENERAL_RETRIEVAL_QA":
            parts = [description] if description else []
            if data_date:
                parts.append(f"데이터 기준: {data_date}")
            if answer_scope:
                parts.append(f"답변 가능 범위: {answer_scope}")
            if parts:
                return "\n".join(parts)
        if must_not_recommend:
            parts = ["현재 데이터 기준으로는 추천을 확정해서 답변할 수 없습니다."]
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

    normalized_doc_ids = normalize_context_doc_ids(row, max_docs=max_docs)
    if must_disclose_limit or must_not_recommend:
        return "\n".join(
            [
                "현재 데이터 기준으로는 답변을 확정할 수 없습니다.",
                "데이터 기준: 제공된 문서 문맥 없음",
                f"row contract top_doc_id: {safe_text(row.get('top_doc_id')) or '없음'}",
                f"row contract cited_doc_ids: {'|'.join(normalized_doc_ids) if normalized_doc_ids else '없음'}",
            ]
        )
    return "제공된 문서 문맥이 없어 답변을 생성하지 못했습니다."


def has_disclosure_phrase(answer: str) -> bool:
    normalized = safe_text(answer)
    return "데이터 기준:" in normalized and "답변 가능 범위:" in normalized


def build_disclosure_footer(row: dict[str, Any], doc_lookup: dict[str, dict[str, str]], *, max_docs: int) -> str:
    docs = [doc for _, doc in resolve_context_docs(row, doc_lookup, max_docs=max_docs)]
    primary = docs[0] if docs else {}
    data_date = safe_text(primary.get("data_date"))
    answer_scope = safe_text(primary.get("answer_scope"))
    footer_lines = []
    footer_lines.append(f"데이터 기준: {data_date or '제공된 문서 기준'}")
    footer_lines.append(f"답변 가능 범위: {answer_scope or '제공된 문서 문맥 범위'}")
    return "\n".join(footer_lines)


def normalize_disclosure_answer(
    answer: str,
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_docs: int,
) -> str:
    must_disclose_limit = safe_text(row.get("must_disclose_limit")) == "Y"
    if not must_disclose_limit:
        return answer
    if has_disclosure_phrase(answer):
        return answer
    footer = build_disclosure_footer(row, doc_lookup, max_docs=max_docs)
    base_answer = safe_text(answer)
    if base_answer:
        return f"{base_answer.rstrip()}\n{footer}"
    return footer


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    max_docs: int,
    max_description_chars: int,
) -> str:
    if not resolve_context_docs(row, doc_lookup, max_docs=max_docs):
        return normalize_disclosure_answer(
            build_fallback_answer(row, doc_lookup, max_docs=max_docs),
            row,
            doc_lookup,
            max_docs=max_docs,
        )

    prompt = build_context_prompt(
        row,
        doc_lookup=doc_lookup,
        tokenizer=tokenizer,
        max_docs=max_docs,
        max_description_chars=max_description_chars,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
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
    answer = decoded or build_fallback_answer(row, doc_lookup, max_docs=max_docs)
    return normalize_disclosure_answer(answer, row, doc_lookup, max_docs=max_docs)


def write_predictions(
    label: str,
    source_df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_path: Path,
    doc_lookup: dict[str, dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    max_docs: int,
    max_description_chars: int,
    stop_signal_path: Path | None,
    checkpoint_every: int,
) -> dict[str, Any]:
    records = []
    total_rows = len(source_df)
    stopped_early = False
    stop_reason = ""

    for row_number, (_, row) in enumerate(source_df.iterrows(), start=1):
        if stop_signal_path and stop_signal_path.exists():
            stopped_early = True
            stop_reason = f"stop signal detected before processing {label} row {row_number}"
            break
        record = row.to_dict()
        record["answer"] = generate_answer(
            model,
            tokenizer,
            row=record,
            doc_lookup=doc_lookup,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_docs=max_docs,
            max_description_chars=max_description_chars,
        )
        records.append(record)
        if checkpoint_every > 0 and (row_number % checkpoint_every == 0 or row_number == total_rows):
            pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "label": label,
        "rows_written": len(records),
        "source_rows": total_rows,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "output_path": str(output_path),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    final_dir = run_dir / "final"
    if not final_dir.exists():
        raise FileNotFoundError(f"Final finetuned model directory does not exist: {final_dir}")

    max_docs, max_description_chars = resolve_prediction_budget(run_dir, args)
    stop_signal_path = Path(args.stop_signal_path) if args.stop_signal_path else run_dir / "prediction_generation.stop"

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

    doc_lookup = load_apartment_doc_lookup()
    training_candidates = load_csv(QA_PREP_DIR / f"training_candidates_{args.model}.csv")
    valid_df = training_candidates[training_candidates["split"] == "valid"].copy()
    grounded_df = load_csv(QA_PREP_DIR / "holdout_grounded_generation.csv")
    safety_df = load_csv(QA_PREP_DIR / "holdout_edge_safety.csv")

    valid_out = run_dir / "valid_predictions.csv"
    grounded_out = run_dir / "grounded_holdout_predictions.csv"
    safety_out = run_dir / "edge_safety_holdout_predictions.csv"
    prediction_progress_path = run_dir / "prediction_progress.json"

    progress: dict[str, Any] = {
        "run_id": args.run_id,
        "stop_signal_path": str(stop_signal_path),
        "splits": [],
    }

    valid_status = write_predictions(
        "valid",
        valid_df,
        model,
        tokenizer,
        valid_out,
        doc_lookup,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_docs=max_docs,
        max_description_chars=max_description_chars,
        stop_signal_path=stop_signal_path,
        checkpoint_every=args.checkpoint_every,
    )
    progress["splits"].append(valid_status)
    prediction_progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    if valid_status["stopped_early"]:
        print(f"Stopped early: {valid_status['stop_reason']}")
        print(f"Partial predictions saved to {valid_out}")
        return

    grounded_status = write_predictions(
        "grounded_holdout",
        grounded_df,
        model,
        tokenizer,
        grounded_out,
        doc_lookup,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_docs=max_docs,
        max_description_chars=max_description_chars,
        stop_signal_path=stop_signal_path,
        checkpoint_every=args.checkpoint_every,
    )
    progress["splits"].append(grounded_status)
    prediction_progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    if grounded_status["stopped_early"]:
        print(f"Stopped early: {grounded_status['stop_reason']}")
        print(f"Partial predictions saved to {grounded_out}")
        return

    safety_status = write_predictions(
        "edge_safety_holdout",
        safety_df,
        model,
        tokenizer,
        safety_out,
        doc_lookup,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_docs=max_docs,
        max_description_chars=max_description_chars,
        stop_signal_path=stop_signal_path,
        checkpoint_every=args.checkpoint_every,
    )
    progress["splits"].append(safety_status)
    prediction_progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    if safety_status["stopped_early"]:
        print(f"Stopped early: {safety_status['stop_reason']}")
        print(f"Partial predictions saved to {safety_out}")
        return

    print(f"Saved valid predictions to {valid_out}")
    print(f"Saved grounded holdout predictions to {grounded_out}")
    print(f"Saved edge safety holdout predictions to {safety_out}")
    print(f"Saved prediction progress to {prediction_progress_path}")


if __name__ == "__main__":
    main()
