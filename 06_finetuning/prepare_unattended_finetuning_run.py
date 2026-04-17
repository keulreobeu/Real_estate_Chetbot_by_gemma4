from __future__ import annotations

import argparse
import ctypes
import gc
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DEFAULT_MODEL_ID,
    PROJECT_ROOT,
    REPORT_DIR,
    build_run_dir,
    get_context_schema_defaults,
    get_default_frozen_paths,
    load_json,
    load_model_config,
    safe_text,
    save_json,
)


RESERVED_RUN_ARTIFACTS = (
    "manifest.json",
    "config.json",
    "train.log",
    "checkpoints",
    "final",
    "training_result.json",
    "valid_predictions.csv",
    "grounded_holdout_predictions.csv",
    "edge_safety_holdout_predictions.csv",
    "valid_eval.json",
    "grounded_holdout_eval.json",
    "edge_safety_holdout_eval.json",
    "post_train_summary.json",
    "notes.md",
)
PREFLIGHT_OWNED_ARTIFACTS = (
    "schema_v1.md",
    "train_contextual.jsonl",
    "valid_contextual.jsonl",
    "context_build_summary.json",
    "preflight_summary.json",
    "launch_commands.md",
)
CONTEXTUAL_ARTIFACTS = (
    "schema_v1.md",
    "train_contextual.jsonl",
    "valid_contextual.jsonl",
    "context_build_summary.json",
)
JSONL_REQUIRED_KEYS = ("instruction", "input", "output")


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def parse_args() -> argparse.Namespace:
    defaults = get_context_schema_defaults()
    parser = argparse.ArgumentParser(
        description="Prepare a stage 06 unattended finetuning run by validating inputs, optional contextual assets, manifest creation, and environment readiness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--context-mode", choices=["contextual", "frozen"], default="contextual")
    parser.add_argument("--max-docs", type=int, default=int(defaults["max_docs"]))
    parser.add_argument("--max-description-chars", type=int, default=int(defaults["max_description_chars"]))
    parser.add_argument("--max-seq-length", type=int, default=int(defaults["max_seq_length"]))
    parser.add_argument("--training-scope", choices=["full", "gates_and_norms"], default="gates_and_norms")
    parser.add_argument("--operator-note", default="unattended finetuning run prepared before training")
    return parser.parse_args()


def resolve_project_local_run_dir(run_id: str, run_dir_arg: str | None) -> Path:
    run_dir = Path(run_dir_arg).resolve() if run_dir_arg else build_run_dir(run_id).resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(
            "Stage 06 unattended tooling currently supports only project-local run directories under the repository root. "
            "External --run-dir paths are not supported without additional implementation."
        )
    return run_dir


def command_to_string(parts: list[str]) -> str:
    rendered: list[str] = []
    for part in parts:
        rendered.append(f'"{part}"' if any(ch.isspace() for ch in part) else part)
    return " ".join(rendered)


def run_checked(parts: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(parts, cwd=str(cwd), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {command_to_string(parts)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return {
        "command": command_to_string(parts),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_run_dir_state(run_dir: Path) -> dict[str, Any]:
    tracked = list(RESERVED_RUN_ARTIFACTS) + list(PREFLIGHT_OWNED_ARTIFACTS)
    existing_artifacts = sorted(name for name in tracked if (run_dir / name).exists())
    reserved = sorted(name for name in RESERVED_RUN_ARTIFACTS if (run_dir / name).exists())
    preflight_owned = sorted(name for name in PREFLIGHT_OWNED_ARTIFACTS if (run_dir / name).exists())
    contextual = sorted(name for name in CONTEXTUAL_ARTIFACTS if (run_dir / name).exists())
    return {
        "run_dir_exists": run_dir.exists(),
        "existing_artifacts": existing_artifacts,
        "reserved_artifacts": reserved,
        "preflight_owned_artifacts": preflight_owned,
        "contextual_artifacts": contextual,
    }


def inspect_run_dir_for_preflight(run_dir: Path, context_mode: str) -> dict[str, Any]:
    state = collect_run_dir_state(run_dir)
    if state["reserved_artifacts"]:
        raise RuntimeError(
            "Run directory is already reserved by completed or in-progress artifacts: "
            f"{state['reserved_artifacts']}. Choose a new run-id."
        )
    if context_mode == "frozen" and state["contextual_artifacts"]:
        raise RuntimeError(
            "Frozen mode cannot reuse a run directory that already contains contextual artifacts: "
            f"{state['contextual_artifacts']}. Remove those artifacts or choose a new run-id."
        )
    return state


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path} line {line_number}: {exc}") from exc
            records.append(payload)
    return records


def validate_jsonl_contract(path: Path, *, require_empty_input: bool) -> dict[str, Any]:
    rows = 0
    empty_instruction_rows: list[int] = []
    empty_output_rows: list[int] = []
    nonempty_input_rows: list[int] = []
    missing_keys_rows: list[dict[str, Any]] = []
    pair_counter: Counter[tuple[str, str]] = Counter()

    for line_number, payload in enumerate(jsonl_records(path), start=1):
        rows += 1
        missing_keys = [key for key in JSONL_REQUIRED_KEYS if key not in payload]
        if missing_keys:
            missing_keys_rows.append({"line": line_number, "missing_keys": missing_keys})
            continue

        instruction = safe_text(payload.get("instruction", ""))
        input_text = safe_text(payload.get("input", ""))
        output = safe_text(payload.get("output", ""))

        if not instruction:
            empty_instruction_rows.append(line_number)
        if not output:
            empty_output_rows.append(line_number)
        if require_empty_input and input_text:
            nonempty_input_rows.append(line_number)

        pair_counter[(instruction, output)] += 1

    if missing_keys_rows:
        raise RuntimeError(f"{path} is missing required JSONL keys: {missing_keys_rows[:10]}")
    if empty_output_rows:
        raise RuntimeError(f"{path} has empty output rows at lines: {empty_output_rows[:10]}")
    if empty_instruction_rows:
        raise RuntimeError(f"{path} has empty instruction rows at lines: {empty_instruction_rows[:10]}")
    if nonempty_input_rows:
        raise RuntimeError(f"{path} has non-empty input rows at lines: {nonempty_input_rows[:10]}")

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": rows,
        "empty_instruction_rows": len(empty_instruction_rows),
        "empty_output_rows": len(empty_output_rows),
        "nonempty_input_rows": len(nonempty_input_rows),
        "pairs": pair_counter,
    }


def build_candidate_pair_counter(training_candidates_df: pd.DataFrame, split: str) -> Counter[tuple[str, str]]:
    split_df = training_candidates_df[training_candidates_df["split"] == split]
    return Counter((safe_text(row.question), safe_text(row.answer)) for row in split_df.itertuples(index=False))


def validate_split_contracts(
    train_jsonl: Path,
    valid_jsonl: Path,
    training_candidates_path: Path,
    dataset_summary_path: Path,
    grounded_holdout_path: Path,
    safety_holdout_path: Path,
) -> dict[str, Any]:
    train_contract = validate_jsonl_contract(train_jsonl, require_empty_input=True)
    valid_contract = validate_jsonl_contract(valid_jsonl, require_empty_input=True)

    training_candidates_df = pd.read_csv(training_candidates_path, encoding="utf-8-sig")
    dataset_summary = load_json(dataset_summary_path)
    grounded_holdout_df = pd.read_csv(grounded_holdout_path, encoding="utf-8-sig")
    safety_holdout_df = pd.read_csv(safety_holdout_path, encoding="utf-8-sig")

    train_split_rows = int((training_candidates_df["split"] == "train").sum())
    valid_split_rows = int((training_candidates_df["split"] == "valid").sum())
    train_split_pairs = build_candidate_pair_counter(training_candidates_df, "train")
    valid_split_pairs = build_candidate_pair_counter(training_candidates_df, "valid")

    expected_train_rows = int(dataset_summary.get("train_rows", 0))
    expected_valid_rows = int(dataset_summary.get("valid_rows", 0))
    expected_grounded_rows = int(dataset_summary.get("holdout_grounded_generation_rows", 0))
    expected_safety_rows = int(dataset_summary.get("holdout_edge_safety_rows", 0))

    mismatches: list[str] = []
    if train_contract["rows"] != expected_train_rows:
        mismatches.append(f"train_jsonl rows {train_contract['rows']} != dataset_summary train_rows {expected_train_rows}")
    if valid_contract["rows"] != expected_valid_rows:
        mismatches.append(f"valid_jsonl rows {valid_contract['rows']} != dataset_summary valid_rows {expected_valid_rows}")
    if train_split_rows != expected_train_rows:
        mismatches.append(f"training_candidates train split {train_split_rows} != dataset_summary train_rows {expected_train_rows}")
    if valid_split_rows != expected_valid_rows:
        mismatches.append(f"training_candidates valid split {valid_split_rows} != dataset_summary valid_rows {expected_valid_rows}")
    if len(grounded_holdout_df) != expected_grounded_rows:
        mismatches.append(
            f"grounded holdout rows {len(grounded_holdout_df)} != dataset_summary holdout_grounded_generation_rows {expected_grounded_rows}"
        )
    if len(safety_holdout_df) != expected_safety_rows:
        mismatches.append(
            f"edge safety holdout rows {len(safety_holdout_df)} != dataset_summary holdout_edge_safety_rows {expected_safety_rows}"
        )
    if train_contract["pairs"] != train_split_pairs:
        mismatches.append("train JSONL content does not match the current training_candidates train split")
    if valid_contract["pairs"] != valid_split_pairs:
        mismatches.append("valid JSONL content does not match the current training_candidates valid split")

    train_bucket_counts = {
        str(key): int(value)
        for key, value in training_candidates_df[training_candidates_df["split"] == "train"]["sft_bucket"].value_counts(dropna=False).to_dict().items()
    }
    valid_bucket_counts = {
        str(key): int(value)
        for key, value in training_candidates_df[training_candidates_df["split"] == "valid"]["sft_bucket"].value_counts(dropna=False).to_dict().items()
    }
    if train_bucket_counts != dataset_summary.get("train_bucket_counts", {}):
        mismatches.append("training_candidates train bucket counts do not match dataset_summary train_bucket_counts")
    if valid_bucket_counts != dataset_summary.get("valid_bucket_counts", {}):
        mismatches.append("training_candidates valid bucket counts do not match dataset_summary valid_bucket_counts")

    if mismatches:
        raise RuntimeError("Input contract mismatches detected:\n- " + "\n- ".join(mismatches))

    return {
        "train_jsonl": {key: value for key, value in train_contract.items() if key != "pairs"},
        "valid_jsonl": {key: value for key, value in valid_contract.items() if key != "pairs"},
        "training_candidates_rows": int(len(training_candidates_df)),
        "training_candidates_train_rows": train_split_rows,
        "training_candidates_valid_rows": valid_split_rows,
        "grounded_holdout_rows": int(len(grounded_holdout_df)),
        "edge_safety_holdout_rows": int(len(safety_holdout_df)),
        "train_bucket_counts": train_bucket_counts,
        "valid_bucket_counts": valid_bucket_counts,
        "content_alignment": {
            "train_matches_candidates": True,
            "valid_matches_candidates": True,
            "frozen_jsonl_input_is_empty": True,
        },
        "dataset_summary": dataset_summary,
    }


def collect_system_memory() -> dict[str, Any]:
    if not hasattr(ctypes, "windll"):
        return {"available": False}
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"available": False}
    return {
        "available": True,
        "total_gb": round(status.ullTotalPhys / (1024**3), 2),
        "free_gb": round(status.ullAvailPhys / (1024**3), 2),
        "memory_load_percent": int(status.dwMemoryLoad),
    }


def collect_environment(run_dir: Path, model_id: str) -> dict[str, Any]:
    model_config = load_model_config(model_id)
    model_source = model_config.get("local_dir") or model_config.get("hf_model_id")
    if not model_source:
        raise RuntimeError(f"Unable to resolve model source for {model_id}")

    model_source_path = Path(model_source)
    model_source_exists = model_source_path.exists() if model_source_path.is_absolute() or model_source.startswith(".") else False

    tokenizer_probe: dict[str, Any]
    try:
        AutoTokenizer.from_pretrained(model_source, local_files_only=True, trust_remote_code=True)
        tokenizer_probe = {"ok": True}
    except Exception as exc:  # pragma: no cover
        tokenizer_probe = {"ok": False, "error": str(exc)}

    model_probe: dict[str, Any]
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model_probe = {"ok": True, "torch_dtype": str(torch_dtype).replace("torch.", "")}
    except Exception as exc:  # pragma: no cover
        model_probe = {
            "ok": False,
            "torch_dtype": str(torch_dtype).replace("torch.", ""),
            "error": str(exc),
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_probe_path = run_dir / ".write_probe.tmp"
    write_probe_path.write_text("ok", encoding="utf-8")
    write_probe_path.unlink()

    disk_usage = shutil.disk_usage(run_dir)
    gpu_info: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        gpu_info.update(
            {
                "device_index": int(device_index),
                "device_name": torch.cuda.get_device_name(device_index),
                "total_memory_gb": round(total_bytes / (1024**3), 2),
                "free_memory_gb": round(free_bytes / (1024**3), 2),
                "reserved_memory_gb": round(torch.cuda.memory_reserved(device_index) / (1024**3), 2),
                "allocated_memory_gb": round(torch.cuda.memory_allocated(device_index) / (1024**3), 2),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cwd": str(PROJECT_ROOT),
        "model_source": model_source,
        "model_source_exists": model_source_exists,
        "tokenizer_probe": tokenizer_probe,
        "model_probe": model_probe,
        "write_probe_target": str(run_dir.relative_to(PROJECT_ROOT)),
        "gpu": gpu_info,
        "system_memory": collect_system_memory(),
        "disk_free_gb": round(disk_usage.free / (1024**3), 2),
        "disk_total_gb": round(disk_usage.total / (1024**3), 2),
        "manual_checks_required": [
            "Disable sleep or hibernation before unattended training.",
            "Prevent automatic restart from Windows Update for the unattended window.",
            "Keep the terminal session pinned to a stable shell or runner.",
        ],
    }


def powershell_python_command(script_path: Path, *extra: str) -> str:
    return command_to_string([sys.executable, str(script_path), *extra])


def next_report_prefix() -> int:
    prefixes: list[int] = []
    for path in REPORT_DIR.glob("*.md"):
        match = re.match(r"^(\d+)_", path.name)
        if match:
            prefixes.append(int(match.group(1)))
    return (max(prefixes) + 1) if prefixes else 1


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_recovery_notes(run_dir: Path) -> dict[str, Any]:
    state = collect_run_dir_state(run_dir)
    run_id_reusable = not bool(state["reserved_artifacts"])
    blocking_reason = ""
    if state["reserved_artifacts"]:
        blocking_reason = f"reserved_run_artifacts_present:{','.join(state['reserved_artifacts'])}"
    return {
        "run_id_reusable": run_id_reusable,
        "current_run_dir_state": state,
        "blocking_reason": blocking_reason,
        "failure_guidance": {
            "environment_failure_before_reservation": "Fix the local environment and rerun the same run-id. No manifest or training reservation should exist yet.",
            "stale_contextual_assets_in_frozen_mode": "Choose a new run-id or remove the contextual preflight-owned artifacts before retrying frozen mode.",
            "reserved_run_artifacts_already_present": "Do not reuse this run-id. Pick a new run-id for a new unattended run.",
        },
    }


def main() -> None:
    args = parse_args()
    run_dir = resolve_project_local_run_dir(args.run_id, args.run_dir)
    run_state_before = inspect_run_dir_for_preflight(run_dir, args.context_mode)

    frozen_paths = get_default_frozen_paths(args.model)
    required_paths = {
        "train_jsonl": frozen_paths["train_file"],
        "valid_jsonl": frozen_paths["valid_file"],
        "training_candidates": frozen_paths["training_candidates_file"],
        "dataset_summary": frozen_paths["dataset_summary_file"],
        "readiness": frozen_paths["readiness_file"],
        "grounded_holdout": frozen_paths["grounded_holdout_file"],
        "edge_safety_holdout": frozen_paths["edge_safety_holdout_file"],
    }
    missing_required = [name for name, path in required_paths.items() if not path.exists()]
    if missing_required:
        raise FileNotFoundError(f"Missing required frozen inputs before preflight: {missing_required}")

    readiness_command = [
        sys.executable,
        str(PROJECT_ROOT / "05_finetuning_prep" / "validate_06_readiness.py"),
        "--model",
        args.model,
    ]
    executed_commands = [run_checked(readiness_command, cwd=PROJECT_ROOT)]
    readiness_payload = load_json(frozen_paths["readiness_file"])
    if readiness_payload.get("verdict") != "GO":
        raise RuntimeError(f"Stage 06 readiness is not GO: {readiness_payload}")

    input_contract = validate_split_contracts(
        frozen_paths["train_file"],
        frozen_paths["valid_file"],
        frozen_paths["training_candidates_file"],
        frozen_paths["dataset_summary_file"],
        frozen_paths["grounded_holdout_file"],
        frozen_paths["edge_safety_holdout_file"],
    )

    contextual_summary: dict[str, Any] | None = None
    if args.context_mode == "contextual":
        contextual_command = [
            sys.executable,
            str(PROJECT_ROOT / "06_finetuning" / "build_contextual_training_view.py"),
            "--run-id",
            args.run_id,
            "--model",
            args.model,
            "--max-docs",
            str(args.max_docs),
            "--max-description-chars",
            str(args.max_description_chars),
            "--max-seq-length",
            str(args.max_seq_length),
        ]
        executed_commands.append(run_checked(contextual_command, cwd=PROJECT_ROOT))
        contextual_summary = load_json(run_dir / "context_build_summary.json")
        defaults = get_context_schema_defaults()
        if not contextual_summary.get("builder_pass"):
            raise RuntimeError("Contextual build summary did not pass builder validation.")
        if contextual_summary.get("schema_status") != "accepted":
            raise RuntimeError("Contextual build summary is not accepted.")
        if contextual_summary.get("validation_version") != defaults["validation_version"]:
            raise RuntimeError(
                "Contextual build validation version mismatch: "
                f"{contextual_summary.get('validation_version')} != {defaults['validation_version']}"
            )

    environment = collect_environment(run_dir, args.model)
    if not environment["tokenizer_probe"]["ok"]:
        raise RuntimeError(f"Tokenizer probe failed: {environment['tokenizer_probe']['error']}")
    if not environment["model_probe"]["ok"]:
        raise RuntimeError(f"Model probe failed: {environment['model_probe']['error']}")
    if not environment["gpu"]["available"]:
        raise RuntimeError("CUDA GPU is not available. Stop before unattended finetuning.")

    manifest_command = [
        sys.executable,
        str(PROJECT_ROOT / "06_finetuning" / "create_run_manifest.py"),
        "--run-id",
        args.run_id,
        "--model",
        args.model,
        "--operator-note",
        args.operator_note,
        "--context-mode",
        args.context_mode,
    ]
    executed_commands.append(run_checked(manifest_command, cwd=PROJECT_ROOT))
    manifest = load_json(run_dir / "manifest.json")
    actual_context_mode = manifest.get("run_local_inputs", {}).get("contextual_selection_mode")
    expected_context_mode = "accepted_contextual" if args.context_mode == "contextual" else "frozen_stage05"
    if actual_context_mode != expected_context_mode:
        raise RuntimeError(f"Manifest selected unexpected input mode: {actual_context_mode} != {expected_context_mode}")

    launch_commands = {
        "train": powershell_python_command(
            PROJECT_ROOT / "06_finetuning" / "train_finetuning_baseline.py",
            "--run-id",
            args.run_id,
            "--model",
            args.model,
            "--max-seq-length",
            str(args.max_seq_length),
            "--training-scope",
            args.training_scope,
        ),
        "predict": powershell_python_command(
            PROJECT_ROOT / "06_finetuning" / "generate_post_train_prediction_sets.py",
            "--run-id",
            args.run_id,
            "--model",
            args.model,
        ),
        "evaluate": powershell_python_command(
            PROJECT_ROOT / "06_finetuning" / "evaluate_post_finetuning_run.py",
            "--run-id",
            args.run_id,
            "--model",
            args.model,
        ),
    }

    recovery_notes = build_recovery_notes(run_dir)
    summary = {
        "run_id": args.run_id,
        "model_id": args.model,
        "path_contract": "project_local_only",
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "context_mode": args.context_mode,
        "training_scope": args.training_scope,
        "max_seq_length": args.max_seq_length,
        "run_dir_state_before": run_state_before,
        "readiness": {
            "path": str(frozen_paths["readiness_file"].relative_to(PROJECT_ROOT)),
            "verdict": readiness_payload.get("verdict"),
            "checks": readiness_payload.get("checks", []),
            "rewrites_canonical_stage_artifact": True,
        },
        "input_contract": input_contract,
        "contextual_summary_path": str((run_dir / "context_build_summary.json").relative_to(PROJECT_ROOT))
        if args.context_mode == "contextual"
        else "",
        "contextual_summary": contextual_summary or {},
        "manifest_path": str((run_dir / "manifest.json").relative_to(PROJECT_ROOT)),
        "manifest_input_mode": actual_context_mode,
        "environment": environment,
        "launch_commands": launch_commands,
        "executed_commands": executed_commands,
        "manual_blockers_if_any": environment["manual_checks_required"],
        "recovery_notes": recovery_notes,
    }

    preflight_summary_path = run_dir / "preflight_summary.json"
    launch_commands_path = run_dir / "launch_commands.md"
    save_json(summary, preflight_summary_path)
    write_markdown(
        launch_commands_path,
        [
            f"# Launch Commands For {args.run_id}",
            "",
            "- Path contract: `project_local_only`",
            f"- Target run directory: `{run_dir.relative_to(PROJECT_ROOT)}`",
            f"- Requested context mode: `{args.context_mode}`",
            f"- Manifest selected input mode: `{actual_context_mode}`",
            "",
            "## Train",
            f"`{launch_commands['train']}`",
            "",
            "## Predict",
            f"`{launch_commands['predict']}`",
            "",
            "## Evaluate",
            f"`{launch_commands['evaluate']}`",
            "",
            "## Manual Checks Before Training",
            *[f"- {item}" for item in environment["manual_checks_required"]],
        ],
    )

    report_prefix = next_report_prefix()
    report_path = REPORT_DIR / f"{report_prefix:02d}_unattended_finetuning_preflight_{args.run_id}.md"
    report_lines = [
        f"# Unattended Finetuning Preflight Report For {args.run_id}",
        "",
        "## Purpose",
        f"- Prepare a new stage 06 unattended finetuning run for `{args.run_id}`.",
        "- Validate readiness, frozen input alignment, run-id hygiene, optional contextual assets, manifest creation, and local environment checks before training.",
        "",
        "## Input Files",
        *(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in required_paths.values()),
        "",
        "## Output Files",
        f"- `{preflight_summary_path.relative_to(PROJECT_ROOT)}`",
        f"- `{launch_commands_path.relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'manifest.json').relative_to(PROJECT_ROOT)}`",
        f"- `{(run_dir / 'config.json').relative_to(PROJECT_ROOT)}`",
    ]
    if args.context_mode == "contextual":
        report_lines.extend(
            [
                f"- `{(run_dir / 'schema_v1.md').relative_to(PROJECT_ROOT)}`",
                f"- `{(run_dir / 'train_contextual.jsonl').relative_to(PROJECT_ROOT)}`",
                f"- `{(run_dir / 'valid_contextual.jsonl').relative_to(PROJECT_ROOT)}`",
                f"- `{(run_dir / 'context_build_summary.json').relative_to(PROJECT_ROOT)}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Original Row And Column Counts",
            f"- `training_candidates_gemma4_2b.csv`: rows={input_contract['training_candidates_rows']}, columns={len(pd.read_csv(frozen_paths['training_candidates_file'], encoding='utf-8-sig').columns)}",
            f"- `holdout_grounded_generation.csv`: rows={input_contract['grounded_holdout_rows']}, columns={len(pd.read_csv(frozen_paths['grounded_holdout_file'], encoding='utf-8-sig').columns)}",
            f"- `holdout_edge_safety.csv`: rows={input_contract['edge_safety_holdout_rows']}, columns={len(pd.read_csv(frozen_paths['edge_safety_holdout_file'], encoding='utf-8-sig').columns)}",
            "",
            "## Final Row And Column Counts",
            f"- `train_gemma4_2b.jsonl`: rows={input_contract['train_jsonl']['rows']}",
            f"- `valid_gemma4_2b.jsonl`: rows={input_contract['valid_jsonl']['rows']}",
        ]
    )
    if contextual_summary:
        report_lines.extend(
            [
                f"- `train_contextual.jsonl`: rows={contextual_summary.get('split_counts', {}).get('train_rows', 0)}",
                f"- `valid_contextual.jsonl`: rows={contextual_summary.get('split_counts', {}).get('valid_rows', 0)}",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Removed Exact Duplicate Count",
            "- Not applicable in this preflight step. No train/valid/holdout source datasets were rewritten.",
            "",
            "## Newly Generated Columns",
            "- None. This step validates contracts and prepares run-local artifacts only.",
            "",
            "## Processing Steps Summary",
            "- Refreshed the stage 06 readiness verdict and rewrote `stage06_readiness_<model>.json` in the canonical stage-05 output directory.",
            "- Verified train/valid JSONL schema plus content-level alignment against the current `training_candidates` train/valid splits.",
            f"- Prepared run-local assets using context mode `{args.context_mode}`.",
            "- Probed both tokenizer load and training-compatible model load before reserving the run with a manifest.",
            f"- Validated write access against the actual target run directory `{run_dir.relative_to(PROJECT_ROOT)}`.",
            "- Stage 06 unattended tooling supports only project-local run directories under the repository root.",
            "- Created a new run manifest, config template, summary, launch commands, and report only after all blocking checks passed.",
            "",
            "## Major Warnings Or Exceptions",
            *[f"- Manual check required: {item}" for item in environment["manual_checks_required"]],
            "",
            "## Recovery Notes",
            f"- Run-id reusable now: `{recovery_notes['run_id_reusable']}`",
            f"- Blocking reason: `{recovery_notes['blocking_reason'] or 'none'}`",
            f"- Current run-dir artifacts: `{', '.join(recovery_notes['current_run_dir_state']['existing_artifacts']) or 'none'}`",
            f"- Validated write target: `{environment['write_probe_target']}`",
            f"- Environment failure before reservation: {recovery_notes['failure_guidance']['environment_failure_before_reservation']}",
            f"- Stale contextual assets in frozen mode: {recovery_notes['failure_guidance']['stale_contextual_assets_in_frozen_mode']}",
            f"- Reserved run artifacts already present: {recovery_notes['failure_guidance']['reserved_run_artifacts_already_present']}",
            "",
            "## Sample Outputs",
            f"- Readiness verdict: `{readiness_payload.get('verdict')}`",
            f"- Manifest input mode: `{actual_context_mode}`",
            f"- GPU available: `{environment['gpu']['available']}`",
            f"- GPU free memory (GB): `{environment['gpu'].get('free_memory_gb', 'n/a')}`",
            f"- Tokenizer probe: `{environment['tokenizer_probe']['ok']}`",
            f"- Model probe: `{environment['model_probe']['ok']}`",
            f"- Train command: `{launch_commands['train']}`",
        ]
    )
    write_markdown(report_path, report_lines)

    print(f"Prepared unattended finetuning run at {run_dir}")
    print(f"Saved preflight summary to {preflight_summary_path}")
    print(f"Saved launch commands to {launch_commands_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
