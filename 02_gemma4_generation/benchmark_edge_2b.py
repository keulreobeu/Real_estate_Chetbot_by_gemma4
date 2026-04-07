from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = Path(__file__).resolve().parent / "run_generation_mvp.py"
LOG_DIR = PROJECT_ROOT / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-row fast-edge benchmark gate for gemma4_2b.")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--backend", choices=["transformers", "llama_cpp"], default="transformers")
    parser.add_argument("--model", default="gemma4_2b")
    parser.add_argument("--avg-threshold-sec", type=float, default=20.0)
    parser.add_argument("--p95-threshold-sec", type=float, default=35.0)
    parser.add_argument("--rph-threshold", type=float, default=180.0)
    parser.add_argument("--warmup-rows", type=int, default=2)
    parser.add_argument("--exclude-worst", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(RUN_SCRIPT),
        "--mode",
        "edge",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--offset",
        str(args.offset),
        "--limit",
        str(args.sample_size),
        "--profile",
        "fast_edge",
        "--no-startup-check",
        "--checkpoint-every",
        "0",
        "--log-every",
        "5",
        "--output-path",
        str(output_path),
    ]


def compute_stats(output_path: Path, warmup_rows: int, exclude_worst: int) -> dict[str, float]:
    df = pd.read_csv(output_path, encoding="utf-8-sig")
    stats_df = pd.DataFrame(
        {
            "latency_sec": pd.to_numeric(df.get("latency_ms"), errors="coerce") / 1000.0,
            "completion_tokens": pd.to_numeric(df.get("completion_tokens"), errors="coerce").fillna(0),
        }
    ).dropna(subset=["latency_sec"])
    if stats_df.empty:
        raise RuntimeError("No latency_ms values were produced by benchmark run.")
    stats_df = stats_df.reset_index(drop=True)
    lat = stats_df["latency_sec"]

    raw_avg_sec = float(lat.mean())
    raw_p50_sec = float(lat.quantile(0.5))
    raw_p95_sec = float(lat.quantile(0.95))
    raw_rph = float(3600.0 / raw_avg_sec) if raw_avg_sec > 0 else 0.0

    scored_df = stats_df.iloc[warmup_rows:].copy() if warmup_rows > 0 else stats_df.copy()
    scored_before_trim = len(scored_df)
    excluded_count = 0
    if exclude_worst > 0 and len(scored_df) > exclude_worst:
        excluded_count = exclude_worst
        scored_df = scored_df.sort_values("latency_sec").iloc[: len(scored_df) - exclude_worst]
    if scored_df.empty:
        scored_df = stats_df.copy()

    scored = scored_df["latency_sec"]

    avg_sec = float(scored.mean())
    p50_sec = float(scored.quantile(0.5))
    p95_sec = float(scored.quantile(0.95))
    rph = float(3600.0 / avg_sec) if avg_sec > 0 else 0.0
    latency_sum = float(scored.sum())
    completion_tokens_sum = float(scored_df["completion_tokens"].sum())
    tokens_per_sec = float(completion_tokens_sum / latency_sum) if latency_sum > 0 else 0.0

    return {
        "rows": int(len(lat)),
        "scored_rows": int(len(scored)),
        "warmup_rows_ignored": int(min(max(warmup_rows, 0), len(lat))),
        "worst_rows_excluded": int(min(excluded_count, scored_before_trim)),
        "raw_avg_sec": raw_avg_sec,
        "raw_p50_sec": raw_p50_sec,
        "raw_p95_sec": raw_p95_sec,
        "raw_rows_per_hour": raw_rph,
        "avg_sec": avg_sec,
        "p50_sec": p50_sec,
        "p95_sec": p95_sec,
        "rows_per_hour": rph,
        "completion_tokens_per_sec": tokens_per_sec,
    }


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer.")
    if args.offset < 0:
        raise ValueError("--offset must be zero or greater.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = LOG_DIR / f"edge_2b_benchmark_{stamp}.csv"
    report_path = LOG_DIR / f"edge_2b_benchmark_{stamp}.json"
    cmd = build_command(args, output_path)

    print("Benchmark command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if args.dry_run:
        print("Dry run only. No benchmark execution performed.")
        raise SystemExit(0)

    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    stats = compute_stats(output_path, warmup_rows=args.warmup_rows, exclude_worst=args.exclude_worst)

    checks = {
        "avg_sec_ok": stats["avg_sec"] <= args.avg_threshold_sec,
        "p95_sec_ok": stats["p95_sec"] <= args.p95_threshold_sec,
        "rows_per_hour_ok": stats["rows_per_hour"] >= args.rph_threshold,
    }
    passed = all(checks.values())

    payload = {
        "benchmark_file": str(output_path),
        "mode": "edge",
        "profile": "fast_edge",
        "sample_size": args.sample_size,
        "offset": args.offset,
        "scoring_policy": {
            "warmup_rows": args.warmup_rows,
            "exclude_worst_rows": args.exclude_worst,
        },
        "thresholds": {
            "avg_sec": args.avg_threshold_sec,
            "p95_sec": args.p95_threshold_sec,
            "rows_per_hour": args.rph_threshold,
        },
        "stats": stats,
        "checks": checks,
        "passed": passed,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Benchmark stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"Gate checks: {json.dumps(checks, ensure_ascii=False)}")
    print(f"Report saved: {report_path}")
    if not passed:
        print("BENCHMARK_GATE_FAIL: Do not start full run until bottleneck is addressed.")
        raise SystemExit(2)

    print("BENCHMARK_GATE_PASS: Safe to start full edge run.")


if __name__ == "__main__":
    main()
