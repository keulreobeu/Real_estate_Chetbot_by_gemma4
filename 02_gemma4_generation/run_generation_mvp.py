from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DEFAULT_BACKEND,
    INPUT_EDGE,
    INPUT_EVAL,
    INPUT_KNOWLEDGE,
    INPUT_MAIN,
    OUTPUT_SOURCE_INDEX,
    append_citation,
    build_prompt,
    fallback_answer,
    get_prediction_output_path,
    infer_used_fields,
    load_csv,
    load_generation_config,
    pick_default_model_id,
    resolve_model_config,
    retrieve_top_k,
    save_csv,
    safe_text,
)
from inference.registry import get_adapter
from query_service import answer_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 02_gemma4_generation MVP pipeline.")
    parser.add_argument("--mode", choices=["eval", "edge"], default="eval")
    parser.add_argument("--backend", choices=["mock", "transformers", "llama_cpp"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", choices=["default", "fast_edge"], default="default")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--retrieval-threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--save-debug-columns", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--stop-signal-path", default=None)
    parser.add_argument("--heartbeat-path", default=None)
    parser.add_argument("--startup-check", dest="startup_check", action="store_true")
    parser.add_argument("--no-startup-check", dest="startup_check", action="store_false")
    parser.add_argument("--startup-check-full", action="store_true")
    parser.set_defaults(startup_check=True)
    return parser.parse_args()


def load_questions(mode: str) -> pd.DataFrame:
    source = INPUT_EVAL if mode == "eval" else INPUT_EDGE
    df, _ = load_csv(source)
    return df


def run_backend(
    backend: str,
    prompt_text: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> Any:
    adapter = get_adapter(backend)
    return adapter.generate(
        prompt_text=prompt_text,
        model_config=model_config,
        generation_config=generation_config,
    )


def run_startup_check(
    backend: str,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    if backend == "mock":
        print("Startup check skipped for mock backend.")
        return {"probe_type": "mock_skip"}

    adapter = get_adapter(backend)
    if hasattr(adapter, "probe_runtime"):
        probe_generation_config = dict(generation_config)
        probe_generation_config["max_output_tokens"] = min(
            int(probe_generation_config.get("max_output_tokens", 32)),
            16,
        )
        probe_generation_config["temperature"] = 0.0
        probe_mode = "full" if bool(generation_config.get("startup_check_full", False)) else "load_only"
        probe_result = adapter.probe_runtime(
            model_config=model_config,
            generation_config=probe_generation_config,
            include_generation=bool(generation_config.get("startup_check_full", False)),
        )
        timing = probe_result.get("timing", {})
        runtime_info = probe_result.get("runtime_info", {})
        print(
            "Startup check passed: "
            f"mode={probe_mode} backend={backend} model_id={model_config.get('model_id')} "
            f"load_runtime_ms={timing.get('load_runtime_ms', 0)} "
            f"processor_load_ms={timing.get('processor_load_ms', 0)} "
            f"model_load_ms={timing.get('model_load_ms', 0)} "
            f"generate_ms={timing.get('generate_ms', 0)} "
            f"device={runtime_info.get('model_device')} "
            f"device_map={runtime_info.get('device_map_requested')}"
        )
        preview = safe_text(probe_result.get("text_preview"))[:60]
        if preview:
            print(f"Startup generation preview: {preview}")
        return {
            "probe_type": probe_mode,
            "timing": timing,
            "runtime_info": runtime_info,
        }

    probe_generation_config = dict(generation_config)
    probe_generation_config["max_output_tokens"] = min(
        int(probe_generation_config.get("max_output_tokens", 32)),
        16,
    )
    probe_generation_config["temperature"] = 0.0
    probe_prompt = "점검용 짧은 응답을 생성해 주세요."
    inference_result = run_backend(
        backend=backend,
        prompt_text=probe_prompt,
        model_config=model_config,
        generation_config=probe_generation_config,
    )
    preview = safe_text(inference_result.text)[:60]
    print(
        "Startup check passed: "
        f"mode=legacy backend={backend} model_id={model_config.get('model_id')} latency_ms={inference_result.latency_ms} "
        f"text_preview={preview}"
    )
    return {"probe_type": "legacy_generate", "timing": {"generate_ms": inference_result.latency_ms}}


def extract_runtime_debug(raw_response: Any) -> dict[str, Any]:
    debug: dict[str, Any] = {
        "load_runtime_ms": 0,
        "processor_load_ms": 0,
        "model_load_ms": 0,
        "prompt_render_ms": 0,
        "input_prepare_ms": 0,
        "to_device_ms": 0,
        "generate_ms": 0,
        "decode_ms": 0,
        "model_device": "",
        "hf_device_map": "",
        "device_map_requested": "",
        "model_source": "",
        "processor_source": "",
        "local_files_only": False,
    }
    if not isinstance(raw_response, dict):
        return debug
    timing = raw_response.get("timing")
    runtime_info = raw_response.get("runtime_info")
    if isinstance(timing, dict):
        debug.update(
            {
                "load_runtime_ms": int(timing.get("load_runtime_ms", 0) or 0),
                "processor_load_ms": int(timing.get("processor_load_ms", 0) or 0),
                "model_load_ms": int(timing.get("model_load_ms", 0) or 0),
                "prompt_render_ms": int(timing.get("prompt_render_ms", 0) or 0),
                "input_prepare_ms": int(timing.get("input_prepare_ms", 0) or 0),
                "to_device_ms": int(timing.get("to_device_ms", 0) or 0),
                "generate_ms": int(timing.get("generate_ms", 0) or 0),
                "decode_ms": int(timing.get("decode_ms", 0) or 0),
            }
        )
    if isinstance(runtime_info, dict):
        debug.update(
            {
                "model_device": safe_text(runtime_info.get("model_device")),
                "hf_device_map": json.dumps(runtime_info.get("hf_device_map"), ensure_ascii=False)
                if runtime_info.get("hf_device_map") is not None
                else "",
                "device_map_requested": safe_text(runtime_info.get("device_map_requested")),
                "model_source": safe_text(runtime_info.get("model_source")),
                "processor_source": safe_text(runtime_info.get("processor_source")),
                "local_files_only": bool(runtime_info.get("local_files_only", False)),
            }
        )
    return debug


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")
    if args.offset < 0:
        raise ValueError("--offset must be zero or greater.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be zero or greater.")
    if args.log_every < 0:
        raise ValueError("--log-every must be zero or greater.")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be a positive integer when provided.")
    if args.temperature is not None and args.temperature < 0:
        raise ValueError("--temperature must be zero or greater when provided.")
    if args.top_p is not None and args.top_p <= 0:
        raise ValueError("--top-p must be greater than zero when provided.")
    if args.repeat_penalty is not None and args.repeat_penalty <= 0:
        raise ValueError("--repeat-penalty must be greater than zero when provided.")
    if args.output_path is not None and not str(args.output_path).strip():
        raise ValueError("--output-path must not be empty when provided.")
    if args.stop_signal_path is not None and not str(args.stop_signal_path).strip():
        raise ValueError("--stop-signal-path must not be empty when provided.")
    if args.heartbeat_path is not None and not str(args.heartbeat_path).strip():
        raise ValueError("--heartbeat-path must not be empty when provided.")

    try:
        generation_config = load_generation_config()
    except Exception as exc:
        raise RuntimeError(
            "Failed to read generation config. Check 02_gemma4_generation/config/generation_defaults.json"
        ) from exc

    if args.profile == "fast_edge":
        if args.mode != "edge":
            raise ValueError("--profile fast_edge is only supported with --mode edge.")
        generation_config = dict(generation_config)
        generation_config["max_output_tokens"] = 64
        generation_config["temperature"] = 0.0
        generation_config["top_p"] = 1.0
        generation_config["repeat_penalty"] = 1.05

    if args.max_output_tokens is not None:
        generation_config["max_output_tokens"] = int(args.max_output_tokens)
    if args.temperature is not None:
        generation_config["temperature"] = float(args.temperature)
    if args.top_p is not None:
        generation_config["top_p"] = float(args.top_p)
    if args.repeat_penalty is not None:
        generation_config["repeat_penalty"] = float(args.repeat_penalty)
    generation_config["startup_check_full"] = bool(args.startup_check_full)

    backend = args.backend or str(generation_config.get("backend", DEFAULT_BACKEND))

    try:
        model_id = args.model or pick_default_model_id()
        model_config = resolve_model_config(model_id)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read model config. Check 02_gemma4_generation/config/models.local.json and model_id."
        ) from exc

    top_k = args.top_k or int(generation_config.get("top_k", 3))
    if args.profile == "fast_edge":
        top_k = min(top_k, 2)
    retrieval_threshold = float(
        args.retrieval_threshold
        if args.retrieval_threshold is not None
        else generation_config.get("retrieval_score_threshold", 1.0)
    )
    fallback_on_low_score = bool(generation_config.get("fallback_on_low_retrieval_score", True))

    if backend != "mock" and str(model_config.get("runtime")) != backend:
        raise RuntimeError(
            f"Model runtime and requested backend mismatch: runtime={model_config.get('runtime')} backend={backend}."
        )

    if not OUTPUT_SOURCE_INDEX.exists():
        raise FileNotFoundError(
            f"Source index not found: {OUTPUT_SOURCE_INDEX}. Run build_generation_assets.py first."
        )
    if not INPUT_MAIN.exists():
        raise FileNotFoundError(f"Main dataset not found: {INPUT_MAIN}")
    if not INPUT_KNOWLEDGE.exists():
        raise FileNotFoundError(f"Knowledge base not found: {INPUT_KNOWLEDGE}")

    if args.startup_check:
        try:
            run_startup_check(
                backend=backend,
                model_config=model_config,
                generation_config=generation_config,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Startup check failed before long run: {exc}. "
                "Use --no-startup-check only when intentionally skipping this probe."
            ) from exc

    source_df, _ = load_csv(OUTPUT_SOURCE_INDEX)
    detailed_df, _ = load_csv(INPUT_MAIN)
    knowledge_df, _ = load_csv(INPUT_KNOWLEDGE)
    questions_df = load_questions(args.mode)

    start = args.offset
    end = None if args.limit is None else start + args.limit
    questions_df = questions_df.iloc[start:end].copy()

    output_path = Path(str(args.output_path)) if args.output_path else get_prediction_output_path(args.mode, model_id)
    stop_signal_path = (
        Path(str(args.stop_signal_path))
        if args.stop_signal_path
        else output_path.parent / f"{output_path.stem}.stop"
    )
    heartbeat_path = (
        Path(str(args.heartbeat_path))
        if args.heartbeat_path
        else output_path.parent / f"{output_path.stem}.heartbeat.json"
    )
    if stop_signal_path.exists():
        print(f"Stop signal file already exists. Run will stop after checkpoint flush: {stop_signal_path}")

    existing_df = pd.DataFrame()
    if (args.append or args.resume) and output_path.exists():
        existing_df, _ = load_csv(output_path)

    if args.resume:
        if output_path.exists() and not existing_df.empty:
            before_count = len(questions_df)
            if "source_row_index" in existing_df.columns:
                done_indices = {
                    int(value)
                    for value in existing_df["source_row_index"].tolist()
                    if pd.notna(value)
                }
                questions_df = questions_df.loc[~questions_df.index.isin(done_indices)].copy()
                skipped_count = before_count - len(questions_df)
                print(
                    "Resume mode enabled (safe index matching): "
                    f"skipped {skipped_count} already-saved rows, remaining {len(questions_df)} rows."
                )
            elif "question" in existing_df.columns:
                done_questions = {safe_text(value) for value in existing_df["question"].tolist() if safe_text(value)}
                questions_df = questions_df[
                    ~questions_df["question"].fillna("").astype(str).str.strip().isin(done_questions)
                ].copy()
                skipped_count = before_count - len(questions_df)
                print(
                    "Resume mode enabled (legacy question matching): "
                    f"skipped {skipped_count} already-saved questions, remaining {len(questions_df)} questions. "
                    "For safer resume, generate outputs with source_row_index."
                )
            else:
                raise RuntimeError(
                    f"--resume requested but existing output has neither source_row_index nor question: {output_path}"
                )
            args.append = True
        else:
            print("Resume mode enabled but no existing output found. Starting fresh for selected slice.")

    if questions_df.empty:
        raise ValueError(
            f"No questions selected for mode={args.mode} with offset={args.offset} and limit={args.limit}."
        )

    if args.checkpoint_every > 0:
        print(f"Checkpoint mode enabled: saving every {args.checkpoint_every} rows.")

    initial_existing_rows = len(existing_df)
    rows: list[dict[str, Any]] = []
    processed_count = 0
    stopped_by_signal = False
    run_started_at = pd.Timestamp.utcnow()

    def write_heartbeat(
        *,
        state: str,
        event: str,
        current_source_row_index: int | None = None,
        checkpoint_saved: bool = False,
    ) -> None:
        payload = {
            "pid": os.getpid(),
            "state": state,
            "event": event,
            "mode": args.mode,
            "backend": backend,
            "model_id": model_id,
            "output_path": str(output_path),
            "heartbeat_path": str(heartbeat_path),
            "stop_signal_path": str(stop_signal_path),
            "last_event_at": pd.Timestamp.utcnow().isoformat(),
            "run_started_at": run_started_at.isoformat(),
            "processed_count": processed_count,
            "selected_rows": int(len(questions_df)),
            "initial_existing_rows": initial_existing_rows,
            "current_total_rows": int(len(existing_df) + len(rows)),
            "current_source_row_index": current_source_row_index,
            "checkpoint_saved": checkpoint_saved,
        }
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def persist_pending_rows(pending_rows: list[dict[str, Any]], checkpoint: bool) -> None:
        nonlocal existing_df
        if not pending_rows:
            return
        pending_df = pd.DataFrame(pending_rows)
        combined_df = pd.concat([existing_df, pending_df], ignore_index=True)
        if "source_row_index" in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["source_row_index"], keep="first").reset_index(drop=True)
            removed = before_dedup - len(combined_df)
            if removed > 0:
                print(f"Deduplicated {removed} rows by source_row_index during save.")
        save_csv(combined_df, output_path)
        existing_df = combined_df
        pending_rows.clear()
        if checkpoint:
            print(f"Checkpoint saved: total_rows={len(existing_df)} path={output_path}")
        write_heartbeat(
            state="running",
            event="checkpoint_saved" if checkpoint else "rows_persisted",
            checkpoint_saved=checkpoint,
        )

    write_heartbeat(state="starting", event="runner_initialized")

    try:
        for source_row_index, question_row in questions_df.iterrows():
            write_heartbeat(
                state="running",
                event="row_started",
                current_source_row_index=int(source_row_index),
            )
            if stop_signal_path.exists():
                print(f"Stop signal detected: {stop_signal_path}")
                persist_pending_rows(rows, checkpoint=True)
                stopped_by_signal = True
                write_heartbeat(
                    state="stopping",
                    event="stop_signal_detected",
                    current_source_row_index=int(source_row_index),
                    checkpoint_saved=True,
                )
                break

            question = safe_text(question_row.get("question"))
            retrieved_df = retrieve_top_k(question, source_df, top_k=top_k)
            cited_doc_ids = [safe_text(value) for value in retrieved_df["문서ID"].tolist() if safe_text(value)]
            top_doc_id = cited_doc_ids[0] if cited_doc_ids else ""
            retrieval_score = float(retrieved_df.iloc[0]["retrieval_score"]) if not retrieved_df.empty else 0.0
            used_fields = infer_used_fields(question)
            prompt_text = ""
            answer_type = ""
            match_status = ""
            query_type = ""
            runtime_debug: dict[str, Any] = {
                "load_runtime_ms": 0,
                "processor_load_ms": 0,
                "model_load_ms": 0,
                "prompt_render_ms": 0,
                "input_prepare_ms": 0,
                "to_device_ms": 0,
                "generate_ms": 0,
                "decode_ms": 0,
                "model_device": "",
                "hf_device_map": "",
                "device_map_requested": "",
                "model_source": "",
                "processor_source": "",
                "local_files_only": False,
            }

            routed_result = answer_query(question, detailed_df=detailed_df, knowledge_df=knowledge_df)
            if routed_result is not None:
                answer = append_citation(routed_result["answer"], routed_result["cited_doc_ids"])
                cited_doc_ids = routed_result["cited_doc_ids"]
                top_doc_id = routed_result["top_doc_id"]
                used_fields = routed_result["used_fields"]
                insufficient_context = bool(routed_result["insufficient_context"])
                finish_reason = "query_contract"
                latency_ms = 0
                raw_response = None
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                answer_type = str(routed_result["answer_type"])
                match_status = str(routed_result["match_status"])
                query_type = str(routed_result["query_type"])
            elif fallback_on_low_score and retrieval_score < retrieval_threshold:
                answer = fallback_answer(cited_doc_ids)
                insufficient_context = True
                finish_reason = "retrieval_threshold"
                latency_ms = 0
                raw_response = None
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                answer_type = "fallback_answer"
                match_status = "UNKNOWN"
                query_type = "GENERAL_RETRIEVAL_QA"
            else:
                prompt_text = build_prompt(question, retrieved_df)
                try:
                    inference_result = run_backend(
                        backend=backend,
                        prompt_text=prompt_text,
                        model_config=model_config,
                        generation_config=generation_config,
                    )
                except FileNotFoundError as exc:
                    runtime = str(model_config.get("runtime"))
                    if runtime == "transformers":
                        raise RuntimeError(
                            f"Transformers model source not found: {exc}. Check local_dir or hf_model_id in models.local.json."
                        ) from exc
                    raise RuntimeError(
                        f"Model file not found: {exc}. Check model_path and file existence in models.local.json."
                    ) from exc
                except ImportError as exc:
                    raise RuntimeError(
                        f"Could not import local inference runtime: {exc}. Check required packages for the backend."
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(f"Local inference execution failed: {exc}") from exc

                if safe_text(inference_result.text):
                    answer = append_citation(safe_text(inference_result.text), cited_doc_ids)
                    insufficient_context = False
                else:
                    answer = fallback_answer(cited_doc_ids)
                    insufficient_context = True
                finish_reason = inference_result.finish_reason
                latency_ms = inference_result.latency_ms
                raw_response = inference_result.raw_response
                runtime_debug = extract_runtime_debug(raw_response)
                token_usage = inference_result.token_usage if isinstance(inference_result.token_usage, dict) else {}
                prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
                total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
                answer_type = "grounded_generation"
                match_status = "UNKNOWN"
                query_type = "GENERAL_RETRIEVAL_QA"

            record: dict[str, Any] = {
                "source_row_index": int(source_row_index),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "match_status": match_status,
                "query_type": query_type,
                "top_doc_id": top_doc_id,
                "cited_doc_ids": "|".join(cited_doc_ids),
                "used_fields": "|".join(used_fields),
                "retrieval_score": retrieval_score,
                "insufficient_context": insufficient_context,
                "backend": backend,
                "model_id": model_id,
                "runtime": backend if backend == "mock" else str(model_config.get("runtime", backend)),
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "load_runtime_ms": runtime_debug["load_runtime_ms"],
                "processor_load_ms": runtime_debug["processor_load_ms"],
                "model_load_ms": runtime_debug["model_load_ms"],
                "prompt_render_ms": runtime_debug["prompt_render_ms"],
                "input_prepare_ms": runtime_debug["input_prepare_ms"],
                "to_device_ms": runtime_debug["to_device_ms"],
                "generate_ms": runtime_debug["generate_ms"],
                "decode_ms": runtime_debug["decode_ms"],
                "model_device": runtime_debug["model_device"],
                "hf_device_map": runtime_debug["hf_device_map"],
                "device_map_requested": runtime_debug["device_map_requested"],
                "model_source": runtime_debug["model_source"],
                "processor_source": runtime_debug["processor_source"],
                "local_files_only": runtime_debug["local_files_only"],
                "prompt_text": prompt_text if args.save_debug_columns else "",
                "raw_response": ("" if raw_response is None else str(raw_response)) if args.save_debug_columns else "",
            }

            if args.mode == "eval":
                record["expected_answer"] = safe_text(question_row.get("expected_answer"))
                record["expected_doc_id"] = safe_text(question_row.get("expected_doc_id"))
                record["expected_answer_type"] = safe_text(question_row.get("expected_answer_type"))
                record["expected_match_status"] = safe_text(question_row.get("expected_match_status"))
                record["must_include"] = safe_text(question_row.get("must_include"))
                record["must_not_include"] = safe_text(question_row.get("must_not_include"))
            else:
                record["expected_doc"] = safe_text(question_row.get("expected_doc"))
                record["expected_field"] = safe_text(question_row.get("expected_field"))
                record["expected_router_type"] = safe_text(question_row.get("expected_router_type"))
                record["expected_match_status"] = safe_text(question_row.get("expected_match_status"))
                record["must_not_recommend"] = safe_text(question_row.get("must_not_recommend"))
                record["must_disclose_limit"] = safe_text(question_row.get("must_disclose_limit"))

            rows.append(record)
            processed_count += 1
            write_heartbeat(
                state="running",
                event="row_completed",
                current_source_row_index=int(source_row_index),
            )

            if args.log_every > 0 and processed_count % args.log_every == 0:
                elapsed_s = max((pd.Timestamp.utcnow() - run_started_at).total_seconds(), 1e-6)
                avg_s = elapsed_s / processed_count
                remaining = len(questions_df) - processed_count
                eta_s = int(avg_s * remaining)
                print(
                    "Progress: "
                    f"{processed_count}/{len(questions_df)} "
                    f"(avg={avg_s:.2f}s/row, eta={eta_s}s)"
                )

            if args.checkpoint_every > 0 and processed_count % args.checkpoint_every == 0:
                persist_pending_rows(rows, checkpoint=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Persisting completed rows before exit.")
        persist_pending_rows(rows, checkpoint=True)
        write_heartbeat(state="stopped", event="keyboard_interrupt", checkpoint_saved=True)
        raise SystemExit(130)

    if not stopped_by_signal:
        persist_pending_rows(rows, checkpoint=False)
    added_rows = len(existing_df) - initial_existing_rows

    if stopped_by_signal:
        print(f"Stopped by signal after saving checkpoint. path={output_path} total_rows={len(existing_df)}")
        if stop_signal_path.exists():
            stop_signal_path.unlink()
            print(f"Stop signal file removed: {stop_signal_path}")
        write_heartbeat(state="stopped", event="stopped_by_signal", checkpoint_saved=True)
        raise SystemExit(0)

    if initial_existing_rows > 0:
        print(f"Appended {added_rows} rows and saved total {len(existing_df)} rows to {output_path}")
    else:
        print(f"Saved {len(existing_df)} rows to {output_path}")
    write_heartbeat(state="completed", event="run_completed")


if __name__ == "__main__":
    main()
