from __future__ import annotations

import argparse

from common import get_compare_report_path, get_metrics_output_path, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Gemma4 generation metrics across two model runs.")
    parser.add_argument("--mode", choices=["eval", "edge"], default="eval")
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--right-model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_metrics = load_json(get_metrics_output_path(args.mode, args.left_model))
    right_metrics = load_json(get_metrics_output_path(args.mode, args.right_model))

    lines = [
        "# 07 Gemma4 Generation Model Comparison",
        "",
        "## Mode",
        "",
        f"- {args.mode}",
        "",
        "## Compared models",
        "",
        f"- left: {args.left_model}",
        f"- right: {args.right_model}",
        "",
        "## Metrics",
        "",
        "| metric | left | right |",
        "| --- | ---: | ---: |",
    ]

    for key in sorted(set(left_metrics) | set(right_metrics)):
        lines.append(f"| {key} | {left_metrics.get(key, '')} | {right_metrics.get(key, '')} |")

    report_path = get_compare_report_path()
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()
