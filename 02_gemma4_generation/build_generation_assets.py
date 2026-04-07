from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (
    INPUT_EDGE,
    INPUT_EVAL,
    INPUT_MAIN,
    OUTPUT_SOURCE_INDEX,
    REQUIRED_EDGE_COLUMNS,
    REQUIRED_EVAL_COLUMNS,
    REQUIRED_MAIN_COLUMNS,
    REPORT_DIR,
    build_search_text,
    load_csv,
    save_csv,
    validate_columns,
)


OUTPUT_REPORT = REPORT_DIR / "06_gemma4_generation_asset_build_report.md"


def main() -> None:
    if not INPUT_MAIN.exists():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {INPUT_MAIN}")
    if not INPUT_EVAL.exists():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {INPUT_EVAL}")
    if not INPUT_EDGE.exists():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {INPUT_EDGE}")

    main_df, main_encoding = load_csv(INPUT_MAIN)
    eval_df, eval_encoding = load_csv(INPUT_EVAL)
    edge_df, edge_encoding = load_csv(INPUT_EDGE)

    validate_columns(main_df, REQUIRED_MAIN_COLUMNS, INPUT_MAIN.name)
    validate_columns(eval_df, REQUIRED_EVAL_COLUMNS, INPUT_EVAL.name)
    validate_columns(edge_df, REQUIRED_EDGE_COLUMNS, INPUT_EDGE.name)

    selected_columns = [
        "문서ID",
        "아파트명",
        "시도",
        "시군구",
        "동",
        "전용면적",
        "공급면적",
        "공급액(만원)",
        "평당_공급액",
        "가장가까운역",
        "거리_m",
        "가장가까운역_호선요약",
        "환승역여부",
        "description",
        "검색키워드",
        "데이터기준일",
        "질의매칭태그",
        "공원_비교요약",
        "병원_비교요약",
        "교통_비교요약",
    ]
    source_index = main_df[selected_columns].copy()
    source_index["검색텍스트"] = source_index.apply(build_search_text, axis=1)
    save_csv(source_index, OUTPUT_SOURCE_INDEX)

    report_lines = [
        "# 06 Gemma4 Generation Asset Build Report",
        "",
        "## Purpose",
        "",
        "Build a retrieval-ready source index for the planned `02_gemma4_generation` stage.",
        "",
        "## Input files",
        "",
        f"- {INPUT_MAIN}",
        f"- {INPUT_EVAL}",
        f"- {INPUT_EDGE}",
        "",
        "## Encodings detected",
        "",
        f"- {INPUT_MAIN.name}: {main_encoding}",
        f"- {INPUT_EVAL.name}: {eval_encoding}",
        f"- {INPUT_EDGE.name}: {edge_encoding}",
        "",
        "## Row and column counts",
        "",
        f"- main dataset: {len(main_df):,} rows, {len(main_df.columns):,} columns",
        f"- eval dataset: {len(eval_df):,} rows, {len(eval_df.columns):,} columns",
        f"- edge dataset: {len(edge_df):,} rows, {len(edge_df.columns):,} columns",
        f"- source index: {len(source_index):,} rows, {len(source_index.columns):,} columns",
        "",
        "## Newly generated columns",
        "",
        "- 검색텍스트",
        "",
        "## Output files",
        "",
        f"- {OUTPUT_SOURCE_INDEX}",
        "",
        "## Sample output",
        "",
        source_index.head(3).to_csv(index=False),
    ]

    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved source index to {OUTPUT_SOURCE_INDEX}")
    print(f"Saved report to {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
