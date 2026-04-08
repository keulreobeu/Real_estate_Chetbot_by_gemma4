from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor edge generation progress and detect stalled runs.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-rows", type=int, default=2000)
    parser.add_argument("--interval-minutes", type=float, default=10.0)
    parser.add_argument("--avg-alert-sec", type=float, default=60.0)
    parser.add_argument("--p95-alert-sec", type=float, default=180.0)
    parser.add_argument("--warning-minutes", type=float, default=20.0)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--window-rows", type=int, default=25)
    parser.add_argument("--max-checks", type=int, default=0)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--main-log-path", default="")
    parser.add_argument("--heartbeat-path", default="")
    parser.add_argument("--benchmark-json-path", default="")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--output-format", choices=["json", "text"], default="json")
    return parser.parse_args()


def read_gpu_util() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:
        return "NA"


def process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return str(pid) == result.stdout.strip()
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def file_age_minutes(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return max((time.time() - path.stat().st_mtime) / 60.0, 0.0)


def parse_csv_stats(path: Path, window_rows: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "rows": 0,
            "max_source_row_index": None,
            "window_avg_sec": 0.0,
            "window_p95_sec": 0.0,
        }

    df = pd.read_csv(path, encoding="utf-8-sig")
    rows = int(len(df))
    max_source_row_index = None
    if "source_row_index" in df.columns:
        source_idx = pd.to_numeric(df["source_row_index"], errors="coerce").dropna()
        if not source_idx.empty:
            max_source_row_index = int(source_idx.max())

    if rows == 0 or "latency_ms" not in df.columns:
        return {
            "rows": rows,
            "max_source_row_index": max_source_row_index,
            "window_avg_sec": 0.0,
            "window_p95_sec": 0.0,
        }

    lat_s = pd.to_numeric(df["latency_ms"], errors="coerce").dropna() / 1000.0
    if lat_s.empty:
        return {
            "rows": rows,
            "max_source_row_index": max_source_row_index,
            "window_avg_sec": 0.0,
            "window_p95_sec": 0.0,
        }

    window = lat_s.tail(max(window_rows, 1))
    return {
        "rows": rows,
        "max_source_row_index": max_source_row_index,
        "window_avg_sec": float(window.mean()),
        "window_p95_sec": float(window.quantile(0.95)),
    }


def read_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def read_log_flags(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "blocked_by_gate": False,
            "last_run_state": "",
            "run_exit_code": None,
        }
    try:
        tail_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
    except Exception:
        return {
            "blocked_by_gate": False,
            "last_run_state": "",
            "run_exit_code": None,
        }

    blocked_by_gate = False
    last_run_state = ""
    run_exit_code = None
    for line in tail_lines:
        if "BENCHMARK_GATE_FAIL" in line or "RUN_STATE: BLOCKED_BY_GATE" in line:
            blocked_by_gate = True
        if "RUN_STATE:" in line:
            last_run_state = line.split("RUN_STATE:", 1)[1].strip()
        if "Run exit_code=" in line:
            try:
                run_exit_code = int(line.rsplit("=", 1)[1].strip())
            except ValueError:
                run_exit_code = None
    return {
        "blocked_by_gate": blocked_by_gate,
        "last_run_state": last_run_state,
        "run_exit_code": run_exit_code,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_csv = Path(args.output_csv)
    main_log_path = Path(args.main_log_path) if args.main_log_path else None
    heartbeat_path = (
        Path(args.heartbeat_path)
        if args.heartbeat_path
        else output_csv.parent / f"{output_csv.stem}.heartbeat.json"
    )
    benchmark_json_path = Path(args.benchmark_json_path) if args.benchmark_json_path else None

    csv_stats = parse_csv_stats(output_csv, window_rows=args.window_rows)
    heartbeat = read_json_file(heartbeat_path)
    benchmark = read_json_file(benchmark_json_path)
    log_flags = read_log_flags(main_log_path)

    heartbeat_age = file_age_minutes(heartbeat_path)
    output_age = file_age_minutes(output_csv)
    log_age = file_age_minutes(main_log_path)
    benchmark_age = file_age_minutes(benchmark_json_path)

    heartbeat_pid = heartbeat.get("pid")
    process_alive = process_exists(int(heartbeat_pid)) if heartbeat_pid is not None else False
    rows = int(csv_stats["rows"])
    remaining_rows = max(args.target_rows - rows, 0)
    output_fresh = output_age is not None and output_age <= args.warning_minutes
    log_fresh = log_age is not None and log_age <= args.warning_minutes
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= args.warning_minutes

    benchmark_passed = benchmark.get("passed")
    blocked_by_gate = bool(log_flags["blocked_by_gate"]) or (
        benchmark_passed is False
        and not process_alive
        and rows < args.target_rows
        and (benchmark_age is None or benchmark_age <= max(args.stall_minutes, args.warning_minutes) * 4)
    )

    verdict = "RUNNING_HEALTHY"
    reasons: list[str] = []
    if rows >= args.target_rows:
        verdict = "COMPLETED"
        reasons.append("target_reached")
    elif blocked_by_gate:
        verdict = "BLOCKED_BY_GATE"
        reasons.append("benchmark_gate_failed")
    elif process_alive:
        if csv_stats["window_avg_sec"] > args.avg_alert_sec:
            reasons.append(f"window_avg_sec>{args.avg_alert_sec}")
        if csv_stats["window_p95_sec"] > args.p95_alert_sec:
            reasons.append(f"window_p95_sec>{args.p95_alert_sec}")
        if (not output_fresh) and (not log_fresh) and (not heartbeat_fresh):
            verdict = "STALL_WARNING"
            reasons.append("all_signals_stale_while_process_alive")
        elif reasons:
            verdict = "RUNNING_SLOW"
    else:
        stale_signals = [
            age for age in (output_age, log_age, heartbeat_age) if age is not None
        ]
        if stale_signals and max(stale_signals) >= args.stall_minutes and rows < args.target_rows:
            verdict = "STALL_CONFIRMED"
            reasons.append("process_missing")
            reasons.append(f"max_staleness_min>={args.stall_minutes}")
        else:
            verdict = "STALL_WARNING"
            reasons.append("process_missing")

    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "rows": rows,
        "target_rows": args.target_rows,
        "remaining_rows": remaining_rows,
        "max_source_row_index": csv_stats["max_source_row_index"],
        "window_avg_sec": round(csv_stats["window_avg_sec"], 3),
        "window_p95_sec": round(csv_stats["window_p95_sec"], 3),
        "output_last_write_age_min": None if output_age is None else round(output_age, 2),
        "main_log_last_write_age_min": None if log_age is None else round(log_age, 2),
        "heartbeat_last_write_age_min": None if heartbeat_age is None else round(heartbeat_age, 2),
        "benchmark_last_write_age_min": None if benchmark_age is None else round(benchmark_age, 2),
        "process_alive": process_alive,
        "heartbeat_pid": heartbeat_pid,
        "heartbeat_state": heartbeat.get("state", ""),
        "heartbeat_event": heartbeat.get("event", ""),
        "benchmark_passed": benchmark_passed,
        "last_run_state": log_flags["last_run_state"],
        "run_exit_code": log_flags["run_exit_code"],
        "output_fresh": output_fresh,
        "log_fresh": log_fresh,
        "heartbeat_fresh": heartbeat_fresh,
        "alert_reasons": reasons,
        "gpu_csv": read_gpu_util(),
        "paths": {
            "output_csv": str(output_csv),
            "main_log_path": "" if main_log_path is None else str(main_log_path),
            "heartbeat_path": str(heartbeat_path),
            "benchmark_json_path": "" if benchmark_json_path is None else str(benchmark_json_path),
        },
    }


def emit(payload: dict[str, Any], output_format: str, log_path: Path | None) -> None:
    if output_format == "text":
        line = (
            "VERDICT: "
            f"{payload['verdict']} | rows={payload['rows']} | max_source_row_index={payload['max_source_row_index']} "
            f"| remaining={payload['remaining_rows']} | process_alive={str(payload['process_alive']).lower()} "
            f"| output_fresh={str(payload['output_fresh']).lower()} | log_fresh={str(payload['log_fresh']).lower()} "
            f"| heartbeat_fresh={str(payload['heartbeat_fresh']).lower()}"
        )
        body = [
            line,
            f"benchmark_passed={payload['benchmark_passed']}",
            f"last_run_state={payload['last_run_state']}",
            f"run_exit_code={payload['run_exit_code']}",
            f"output_last_write_age_min={payload['output_last_write_age_min']}",
            f"main_log_last_write_age_min={payload['main_log_last_write_age_min']}",
            f"heartbeat_last_write_age_min={payload['heartbeat_last_write_age_min']}",
            f"window_avg_sec={payload['window_avg_sec']}",
            f"window_p95_sec={payload['window_p95_sec']}",
            f"alert_reasons={'|'.join(payload['alert_reasons'])}",
        ]
        text = "\n".join(body)
    else:
        text = json.dumps(payload, ensure_ascii=False)

    print(text)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(text + "\n")


def main() -> None:
    args = parse_args()
    if args.interval_minutes <= 0:
        raise ValueError("--interval-minutes must be greater than zero.")
    if args.target_rows <= 0:
        raise ValueError("--target-rows must be greater than zero.")
    if args.warning_minutes <= 0:
        raise ValueError("--warning-minutes must be greater than zero.")
    if args.stall_minutes <= 0:
        raise ValueError("--stall-minutes must be greater than zero.")
    if args.window_rows <= 0:
        raise ValueError("--window-rows must be greater than zero.")

    log_path = Path(args.log_path) if args.log_path else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    checks_done = 0
    while True:
        checks_done += 1
        payload = build_payload(args)
        emit(payload, output_format=args.output_format, log_path=log_path)

        if args.one_shot or payload["verdict"] == "COMPLETED":
            return
        if args.max_checks > 0 and checks_done >= args.max_checks:
            return
        time.sleep(int(args.interval_minutes * 60))


if __name__ == "__main__":
    main()
