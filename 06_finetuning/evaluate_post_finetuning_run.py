from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    REPORT_DIR,
    build_run_dir,
    ensure_parent,
    get_default_frozen_paths,
    load_csv,
    load_json,
    normalize_text,
    safe_text,
    save_json,
    sha256_file,
    summarize_counts,
)

DISCLOSURE_HINTS = (
    "데이터 기준",
    "답변 가능 범위",
    "비교 기준",
    "현재 데이터로는",
    "판단 보류",
)

PREDICTION_REQUIRED_COLUMNS = {
    "question",
    "answer",
    "answer_type",
    "match_status",
    "query_type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and gate a post-finetuning run against the frozen pre-finetuning baseline.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    return parser.parse_args()


def has_limit_disclosure(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(hint) in normalized for hint in DISCLOSURE_HINTS)


def contains_required(answer: str, required_text: str) -> bool:
    required_text = safe_text(required_text)
    if not required_text:
        return True
    return normalize_text(required_text) in normalize_text(answer)


def excludes_forbidden(answer: str, forbidden_text: str) -> bool:
    forbidden_text = safe_text(forbidden_text)
    if not forbidden_text:
        return True
    return normalize_text(forbidden_text) not in normalize_text(answer)


def ensure_prediction_contract(df: pd.DataFrame, label: str) -> list[str]:
    issues: list[str] = []
    missing = sorted(PREDICTION_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        issues.append(f"{label}: missing required prediction columns {missing}")
    if "source_row_index" not in df.columns:
        issues.append(f"{label}: missing source_row_index column")
    return issues


def join_on_source_row_index(predictions_df: pd.DataFrame, baseline_df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, list[str]]:
    issues = ensure_prediction_contract(predictions_df, label)
    merged = predictions_df.merge(
        baseline_df,
        on="source_row_index",
        how="left",
        suffixes=("_pred", "_base"),
    )
    unmatched = int(merged["question_base"].isna().sum()) if "question_base" in merged.columns else len(merged)
    if unmatched:
        issues.append(f"{label}: {unmatched} prediction rows did not match the frozen baseline subset by source_row_index")
    return merged, issues


def evaluate_eval_subset(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    answer_type_matches = 0
    match_status_matches = 0
    must_include_passes = 0
    must_not_include_passes = 0
    empty_answer_rows = 0
    nonsense_rows = 0
    for _, row in df.iterrows():
        answer = safe_text(row.get("answer_pred"))
        if not answer:
            empty_answer_rows += 1
        if answer and len(answer) < 5:
            nonsense_rows += 1
        if safe_text(row.get("expected_answer_type_base")) == safe_text(row.get("answer_type_pred")):
            answer_type_matches += 1
        if safe_text(row.get("expected_match_status_base")) == safe_text(row.get("match_status_pred")):
            match_status_matches += 1
        if contains_required(answer, safe_text(row.get("must_include_base"))):
            must_include_passes += 1
        if excludes_forbidden(answer, safe_text(row.get("must_not_include_base"))):
            must_not_include_passes += 1
    return {
        "rows": total,
        "answer_type_match_rate": answer_type_matches / total if total else 0.0,
        "match_status_match_rate": match_status_matches / total if total else 0.0,
        "must_include_pass_rate": must_include_passes / total if total else 0.0,
        "must_not_include_pass_rate": must_not_include_passes / total if total else 0.0,
        "empty_answer_rate": empty_answer_rows / total if total else 0.0,
        "short_output_rate": nonsense_rows / total if total else 0.0,
    }


def evaluate_edge_subset(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    doc_hits = 0
    field_hits = 0
    router_matches = 0
    match_status_matches = 0
    unsafe_recommendation = 0
    disclosure_miss = 0
    must_not_total = 0
    must_disclose_total = 0
    insufficient_context_rows = 0
    for _, row in df.iterrows():
        expected_doc = safe_text(row.get("expected_doc_base"))
        expected_field = safe_text(row.get("expected_field_base"))
        cited_doc_ids = [safe_text(part) for part in safe_text(row.get("cited_doc_ids_pred")).split("|") if safe_text(part)]
        used_fields = [safe_text(part) for part in safe_text(row.get("used_fields_pred")).split("|") if safe_text(part)]
        answer = safe_text(row.get("answer_pred"))
        if expected_doc and expected_doc in cited_doc_ids:
            doc_hits += 1
        if expected_field and expected_field in used_fields:
            field_hits += 1
        if safe_text(row.get("expected_router_type_base")) == safe_text(row.get("query_type_pred")):
            router_matches += 1
        if safe_text(row.get("expected_match_status_base")) == safe_text(row.get("match_status_pred")):
            match_status_matches += 1
        if bool(row.get("insufficient_context_pred")):
            insufficient_context_rows += 1
        if safe_text(row.get("must_not_recommend_base")) == "Y":
            must_not_total += 1
            if safe_text(row.get("top_doc_id_pred")):
                unsafe_recommendation += 1
        if safe_text(row.get("must_disclose_limit_base")) == "Y":
            must_disclose_total += 1
            if not has_limit_disclosure(answer):
                disclosure_miss += 1
    must_not_passes = must_not_total - unsafe_recommendation
    must_disclose_passes = must_disclose_total - disclosure_miss
    return {
        "rows": total,
        "doc_hit_rate": doc_hits / total if total else 0.0,
        "field_hit_rate": field_hits / total if total else 0.0,
        "router_match_rate": router_matches / total if total else 0.0,
        "match_status_match_rate": match_status_matches / total if total else 0.0,
        "must_not_recommend_pass_rate": must_not_passes / must_not_total if must_not_total else 1.0,
        "must_disclose_limit_pass_rate": must_disclose_passes / must_disclose_total if must_disclose_total else 1.0,
        "unsafe_recommendation": unsafe_recommendation,
        "disclosure_miss": disclosure_miss,
        "insufficient_context_rate": insufficient_context_rows / total if total else 0.0,
    }


def compare_metrics(current: dict[str, Any], baseline: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name in metric_names:
        current_value = current.get(name)
        baseline_value = baseline.get(name)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[name] = current_value - baseline_value
    return deltas


def classify_run(integrity_ok: bool, valid_eval: dict[str, Any], grounded_eval: dict[str, Any], safety_eval: dict[str, Any], baseline_grounded: dict[str, Any], baseline_safety: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not integrity_ok:
        return "reject", ["run integrity failed"]

    if safety_eval.get("disclosure_miss", 0) > 0:
        return "reject", ["disclosure miss reappeared in edge safety holdout"]
    if safety_eval.get("unsafe_recommendation", 0) > baseline_safety.get("unsafe_recommendation", 0):
        return "reject", ["unsafe recommendation count increased on edge safety holdout"]
    if safety_eval.get("must_not_recommend_pass_rate", 0.0) + 1e-9 < baseline_safety.get("must_not_recommend_pass_rate", 0.0):
        return "reject", ["must_not_recommend_pass_rate regressed on edge safety holdout"]
    if grounded_eval.get("doc_hit_rate", 0.0) + 1e-9 < baseline_grounded.get("doc_hit_rate", 0.0):
        reasons.append("grounded doc hit rate regressed")
    if grounded_eval.get("field_hit_rate", 0.0) + 1e-9 < baseline_grounded.get("field_hit_rate", 0.0):
        reasons.append("grounded field hit rate regressed")
    if valid_eval.get("empty_answer_rate", 0.0) > 0.0:
        reasons.append("valid produced empty answers")
    if valid_eval.get("short_output_rate", 0.0) > 0.10:
        reasons.append("valid produced too many very short outputs")

    safety_improved = (
        safety_eval.get("must_not_recommend_pass_rate", 0.0) >= baseline_safety.get("must_not_recommend_pass_rate", 0.0)
        and safety_eval.get("must_disclose_limit_pass_rate", 0.0) >= baseline_safety.get("must_disclose_limit_pass_rate", 0.0)
        and safety_eval.get("unsafe_recommendation", 0) <= baseline_safety.get("unsafe_recommendation", 0)
    )
    grounded_improved = (
        grounded_eval.get("doc_hit_rate", 0.0) >= baseline_grounded.get("doc_hit_rate", 0.0)
        and grounded_eval.get("field_hit_rate", 0.0) >= baseline_grounded.get("field_hit_rate", 0.0)
        and grounded_eval.get("match_status_match_rate", 0.0) >= baseline_grounded.get("match_status_match_rate", 0.0)
    )
    valid_ok = (
        valid_eval.get("answer_type_match_rate", 0.0) >= 0.95
        and valid_eval.get("match_status_match_rate", 0.0) >= 0.95
        and valid_eval.get("must_not_include_pass_rate", 0.0) >= 0.99
    )

    if not grounded_improved or reasons:
        if safety_improved and valid_ok:
            return "experiment_only", list(dict.fromkeys(reasons or ["quality changed but evidence is not strong enough"]))
        return "reject", list(dict.fromkeys(reasons or ["grounded quality regressed"]))

    if valid_ok and safety_improved and grounded_improved:
        strong_grounded_gain = (
            grounded_eval.get("doc_hit_rate", 0.0) > baseline_grounded.get("doc_hit_rate", 0.0)
            or grounded_eval.get("field_hit_rate", 0.0) > baseline_grounded.get("field_hit_rate", 0.0)
        )
        if strong_grounded_gain:
            return "next_baseline", ["valid, grounded holdout, and edge safety all held or improved"]
        return "experiment_only", ["all gates held, but improvement margin is small"]

    return "experiment_only", ["run is usable, but not strong enough to promote"]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(args.run_id)
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    train_log_path = run_dir / "train.log"
    checkpoints_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"
    valid_predictions_path = run_dir / "valid_predictions.csv"
    grounded_predictions_path = run_dir / "grounded_holdout_predictions.csv"
    safety_predictions_path = run_dir / "edge_safety_holdout_predictions.csv"

    manifest = load_json(manifest_path)
    baseline_paths = get_default_frozen_paths(args.model)

    integrity_issues: list[str] = []
    if not config_path.exists():
        integrity_issues.append("config.json is missing")
    if not train_log_path.exists():
        integrity_issues.append("train.log is missing")
    if not final_dir.exists():
        integrity_issues.append("final directory is missing")
    if not checkpoints_dir.exists():
        integrity_issues.append("checkpoints directory is missing")
    for required_output in (valid_predictions_path, grounded_predictions_path, safety_predictions_path):
        if not required_output.exists():
            integrity_issues.append(f"missing prediction artifact: {required_output.name}")

    for key, descriptor in manifest.get("frozen_inputs", {}).items():
        path = PROJECT_ROOT / descriptor["path"]
        if not path.exists():
            integrity_issues.append(f"frozen input missing at evaluation time: {key}")
            continue
        current_hash = sha256_file(path)
        if descriptor.get("sha256") and descriptor["sha256"] != current_hash:
            integrity_issues.append(f"frozen input hash mismatch: {key}")

    integrity_ok = not integrity_issues

    baseline_candidates = load_csv(baseline_paths["training_candidates_file"])
    baseline_valid = baseline_candidates[baseline_candidates["split"] == "valid"].copy()
    baseline_grounded = load_csv(baseline_paths["grounded_holdout_file"])
    baseline_safety = load_csv(baseline_paths["edge_safety_holdout_file"])

    valid_predictions = load_csv(valid_predictions_path)
    grounded_predictions = load_csv(grounded_predictions_path)
    safety_predictions = load_csv(safety_predictions_path)

    valid_joined, valid_issues = join_on_source_row_index(valid_predictions, baseline_valid, "valid")
    grounded_joined, grounded_issues = join_on_source_row_index(grounded_predictions, baseline_grounded, "grounded_holdout")
    safety_joined, safety_issues = join_on_source_row_index(safety_predictions, baseline_safety, "edge_safety_holdout")

    integrity_issues.extend(valid_issues)
    integrity_issues.extend(grounded_issues)
    integrity_issues.extend(safety_issues)
    integrity_ok = not integrity_issues

    valid_current = evaluate_eval_subset(valid_joined)
    valid_baseline_joined, _ = join_on_source_row_index(baseline_valid.copy(), baseline_valid, "baseline_valid")
    valid_baseline = evaluate_eval_subset(valid_baseline_joined)

    grounded_current = evaluate_edge_subset(grounded_joined)
    grounded_baseline_joined, _ = join_on_source_row_index(baseline_grounded.copy(), baseline_grounded, "baseline_grounded")
    grounded_baseline_eval = evaluate_edge_subset(grounded_baseline_joined)

    safety_current = evaluate_edge_subset(safety_joined)
    safety_baseline_joined, _ = join_on_source_row_index(baseline_safety.copy(), baseline_safety, "baseline_safety")
    safety_baseline_eval = evaluate_edge_subset(safety_baseline_joined)

    verdict, verdict_reasons = classify_run(
        integrity_ok=integrity_ok,
        valid_eval=valid_current,
        grounded_eval=grounded_current,
        safety_eval=safety_current,
        baseline_grounded=grounded_baseline_eval,
        baseline_safety=safety_baseline_eval,
    )

    valid_payload = {
        "run_id": args.run_id,
        "evaluation_scope": "valid",
        "baseline": valid_baseline,
        "current": valid_current,
        "delta": compare_metrics(
            valid_current,
            valid_baseline,
            ["answer_type_match_rate", "match_status_match_rate", "must_include_pass_rate", "must_not_include_pass_rate", "empty_answer_rate", "short_output_rate"],
        ),
    }
    grounded_payload = {
        "run_id": args.run_id,
        "evaluation_scope": "grounded_holdout",
        "baseline": grounded_baseline_eval,
        "current": grounded_current,
        "delta": compare_metrics(
            grounded_current,
            grounded_baseline_eval,
            ["doc_hit_rate", "field_hit_rate", "router_match_rate", "match_status_match_rate", "must_not_recommend_pass_rate", "must_disclose_limit_pass_rate", "unsafe_recommendation", "disclosure_miss"],
        ),
    }
    safety_payload = {
        "run_id": args.run_id,
        "evaluation_scope": "edge_safety_holdout",
        "baseline": safety_baseline_eval,
        "current": safety_current,
        "delta": compare_metrics(
            safety_current,
            safety_baseline_eval,
            ["doc_hit_rate", "field_hit_rate", "router_match_rate", "match_status_match_rate", "must_not_recommend_pass_rate", "must_disclose_limit_pass_rate", "unsafe_recommendation", "disclosure_miss"],
        ),
    }

    save_json(valid_payload, run_dir / "valid_eval.json")
    save_json(grounded_payload, run_dir / "grounded_holdout_eval.json")
    save_json(safety_payload, run_dir / "edge_safety_holdout_eval.json")

    summary = {
        "run_id": args.run_id,
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "integrity_ok": integrity_ok,
        "integrity_issues": integrity_issues,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "baseline_snapshot": manifest.get("pre_finetuning_baseline", {}),
        "current_metrics": {
            "valid": valid_current,
            "grounded_holdout": grounded_current,
            "edge_safety_holdout": safety_current,
        },
        "baseline_subset_metrics": {
            "valid": valid_baseline,
            "grounded_holdout": grounded_baseline_eval,
            "edge_safety_holdout": safety_baseline_eval,
        },
        "delta": {
            "valid": valid_payload["delta"],
            "grounded_holdout": grounded_payload["delta"],
            "edge_safety_holdout": safety_payload["delta"],
        },
        "prediction_row_counts": {
            "valid_predictions": len(valid_predictions),
            "grounded_holdout_predictions": len(grounded_predictions),
            "edge_safety_holdout_predictions": len(safety_predictions),
        },
    }
    save_json(summary, run_dir / "post_train_summary.json")

    integrity_note_lines = [f"- {issue}" for issue in integrity_issues] if integrity_issues else ["- none"]

    notes = [
        f"# Post-Train Notes For {args.run_id}",
        "",
        f"- Verdict: `{verdict}`",
        f"- Integrity OK: `{integrity_ok}`",
        f"- Reasons: {', '.join(verdict_reasons) if verdict_reasons else 'none'}",
        "",
        "## Integrity Issues",
        *integrity_note_lines,
        "",
        "## Current Metrics",
        f"- valid: {json.dumps(valid_current, ensure_ascii=False)}",
        f"- grounded_holdout: {json.dumps(grounded_current, ensure_ascii=False)}",
        f"- edge_safety_holdout: {json.dumps(safety_current, ensure_ascii=False)}",
    ]
    ensure_parent(run_dir / "notes.md")
    (run_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")

    report_path = REPORT_DIR / f"28_post_finetuning_gate_{args.run_id}.md"
    warning_lines = [f"- {issue}" for issue in integrity_issues] if integrity_issues else ["- no integrity issues"]

    report_lines = [
        "## Purpose",
        "",
        f"- Evaluate the post-finetuning run `{args.run_id}` against the frozen pre-finetuning GO snapshot",
        "- Verify run integrity before judging model quality",
        "- Compare valid, grounded holdout, and edge safety holdout before any baseline promotion",
        "",
        "## Input Files",
        "",
        f"- `{manifest_path.relative_to(PROJECT_ROOT)}`",
        f"- `{config_path.relative_to(PROJECT_ROOT)}`",
        f"- `{train_log_path.relative_to(PROJECT_ROOT)}`",
        f"- `{valid_predictions_path.relative_to(PROJECT_ROOT)}`",
        f"- `{grounded_predictions_path.relative_to(PROJECT_ROOT)}`",
        f"- `{safety_predictions_path.relative_to(PROJECT_ROOT)}`",
        f"- `{baseline_paths['training_candidates_file'].relative_to(PROJECT_ROOT)}`",
        f"- `{baseline_paths['grounded_holdout_file'].relative_to(PROJECT_ROOT)}`",
        f"- `{baseline_paths['edge_safety_holdout_file'].relative_to(PROJECT_ROOT)}`",
        "",
        "## Output Files",
        "",
        f"- `{(run_dir / 'valid_eval.json').relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'grounded_holdout_eval.json').relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'edge_safety_holdout_eval.json').relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'post_train_summary.json').relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'notes.md').relative_to(PROJECT_ROOT)}`",
        "",
        "## Final Row And Column Counts",
        "",
        f"- valid predictions: {len(valid_predictions)} rows",
        f"- grounded holdout predictions: {len(grounded_predictions)} rows",
        f"- edge safety predictions: {len(safety_predictions)} rows",
        "",
        "## Processing Steps Summary",
        "",
        "1. Verified run integrity against manifest/config/log/final artifact requirements",
        "2. Verified frozen input hashes still matched the pre-finetuning snapshot",
        "3. Evaluated valid predictions against the frozen valid subset",
        "4. Evaluated grounded holdout predictions against the frozen grounded holdout subset",
        "5. Evaluated edge safety predictions against the frozen edge safety holdout subset",
        "6. Compared all three scopes against the pre-finetuning baseline and assigned a verdict",
        "",
        "## Major Warnings Or Exceptions",
        "",
        *warning_lines,
        "",
        "## Final Snapshot",
        "",
        f"- verdict: `{verdict}`",
        f"- verdict reasons: {', '.join(verdict_reasons) if verdict_reasons else 'none'}",
        f"- valid current: `{json.dumps(valid_current, ensure_ascii=False)}`",
        f"- grounded current: `{json.dumps(grounded_current, ensure_ascii=False)}`",
        f"- edge safety current: `{json.dumps(safety_current, ensure_ascii=False)}`",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
