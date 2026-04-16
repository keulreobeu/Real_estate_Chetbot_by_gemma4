from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
QA_DIR = DATA_DIR / "qa"
REPORT_DIR = PROJECT_ROOT / "00_Report"
INPUT_FILE = DATA_DIR / "apartment_chatbot_v3.csv"
OUTPUT_QUESTIONS = QA_DIR / "edge_case_questions.csv"
OUTPUT_EVAL = QA_DIR / "edge_case_eval.csv"
OUTPUT_REPORT = REPORT_DIR / "04_edge_question_report.md"

RANDOM_SEED = 42
TOTAL_TARGET = 2000
TYPE_TARGETS = {
    "condition": 250,
    "comparison": 250,
    "multi_condition": 250,
    "region": 900,
    "vague": 175,
    "colloquial": 175,
}
ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV를 읽지 못했습니다: {path} ({last_error})")


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "None", "NULL", "null"}:
        return True
    return False


def q(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| 데이터 없음 |\n| --- |"
    safe = df.fillna("").copy()
    headers = [str(col) for col in safe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in safe.iterrows():
        values = [str(v).replace("\n", " ").replace("|", "/") for v in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def add_record(
    store: list[dict[str, str]],
    seen_questions: set[str],
    question: str | None,
    qtype: str,
    expected_doc: str | None,
    expected_field: str,
    expected_router_type: str,
    expected_match_status: str,
    must_not_recommend: str = "N",
    must_disclose_limit: str = "N",
) -> bool:
    if not question:
        return False
    question = question.strip()
    if not question or question in seen_questions:
        return False
    seen_questions.add(question)
    store.append(
        {
            "question": question,
            "type": qtype,
            "expected_doc": expected_doc or "",
            "expected_field": expected_field,
            "expected_router_type": expected_router_type,
            "expected_match_status": expected_match_status,
            "must_not_recommend": must_not_recommend,
            "must_disclose_limit": must_disclose_limit,
        }
    )
    return True


def pick_doc(df: pd.DataFrame, sort_by: str, ascending: bool = True) -> str | None:
    if df.empty:
        return None
    if sort_by not in df.columns:
        return q(df.iloc[0].get("문서ID"))
    subset = df[df[sort_by].notna()] if df[sort_by].notna().any() else df
    if subset.empty:
        return None
    picked = subset.sort_values(sort_by, ascending=ascending).iloc[0]
    return q(picked.get("문서ID"))


def top_values(df: pd.DataFrame, column: str, limit: int) -> list[str]:
    if column not in df.columns:
        return []
    return df[column].dropna().astype(str).value_counts().index.tolist()[:limit]


def build_condition_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = top_values(df, "시군구", 12)
    top_dongs = top_values(df, "동", 12)
    distances = [300, 500, 700, 1000]

    for region in top_regions:
        region_df = df[df["시군구"] == region]
        for dist in distances:
            subset = region_df[region_df["거리_m"].notna() & (region_df["거리_m"] <= dist)]
            add_record(
                records,
                seen_questions,
                f"{region}에서 지하철 {dist}m 이내 아파트 추천해줘",
                "condition",
                pick_doc(subset, "거리_m", True),
                "거리_m",
                "RECOMMEND_STRUCTURED",
                "EXACT_MATCH" if not subset.empty else "NO_MATCH",
                must_not_recommend="Y" if subset.empty else "N",
            )
        cheap_subset = region_df[region_df["가격요약"].fillna("").str.contains("합리적")]
        add_record(
            records,
            seen_questions,
            f"{region}에서 가격 괜찮은 아파트 추천해줘",
            "condition",
            pick_doc(cheap_subset, "평당_공급액", True),
            "가격요약",
            "RECOMMEND_STRUCTURED",
            "EXACT_MATCH" if not cheap_subset.empty else "NO_MATCH",
            must_not_recommend="Y" if cheap_subset.empty else "N",
        )

    for dong in top_dongs:
        dong_df = df[df["동"] == dong]
        hospital_subset = dong_df[dong_df["병원_접근지표"].notna()]
        add_record(
            records,
            seen_questions,
            f"{dong}에서 병원 접근 좋은 아파트 추천해줘",
            "condition",
            pick_doc(hospital_subset, "병원_접근지표", False),
            "병원_접근지표",
            "RECOMMEND_STRUCTURED",
            "EXACT_MATCH" if not hospital_subset.empty else "UNKNOWN",
            must_not_recommend="Y" if hospital_subset.empty else "N",
        )

    no_match_questions = [
        "서울에서 1억 이하이면서 지하철 100m 이내 아파트 추천해줘",
        "송파구에서 2억 이하 대단지 신축 아파트 추천해줘",
    ]
    for question in no_match_questions:
        add_record(
            records,
            seen_questions,
            question,
            "condition",
            "",
            "거리_m",
            "RECOMMEND_STRUCTURED",
            "NO_MATCH",
            must_not_recommend="Y",
        )

    return records[:target]


def build_comparison_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = top_values(df, "시군구", 15)

    for region in top_regions:
        region_df = df[df["시군구"] == region]
        specs = [
            ("공원 가까운 아파트 비교해줘", "공원_접근지표", "공원_비교요약"),
            ("병원 접근성 좋은 아파트 비교해줘", "병원_접근지표", "병원_비교요약"),
            ("지하철 접근성 좋은 아파트 비교해줘", "지하철_접근지표", "교통_비교요약"),
        ]
        for suffix, score_column, field in specs:
            subset = region_df[region_df[score_column].notna()] if score_column in region_df.columns else region_df.iloc[0:0]
            add_record(
                records,
                seen_questions,
                f"{region}에서 {suffix}",
                "comparison",
                pick_doc(subset, score_column, False),
                field,
                "RECOMMEND_COMPARATIVE",
                "EXACT_MATCH" if not subset.empty else "UNKNOWN",
                must_not_recommend="Y" if subset.empty else "N",
                must_disclose_limit="N",
            )

    unsupported_questions = [
        "아이 키우기 좋은 아파트 추천해줘",
        "살기 좋은 아파트 비교해줘",
        "미래 가치 높은 아파트 추천해줘",
    ]
    for question in unsupported_questions:
        add_record(
            records,
            seen_questions,
            question,
            "comparison",
            "",
            "지원 범위",
            "RECOMMEND_COMPARATIVE",
            "UNKNOWN",
            must_not_recommend="Y",
            must_disclose_limit="Y",
        )

    return records[:target]


def build_multi_condition_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = top_values(df, "시군구", 14)
    bands = top_values(df, "면적대", 5)

    for region in top_regions:
        for band in bands:
            subset = df[
                (df["시군구"] == region)
                & (df["면적대"] == band)
                & df["거리_m"].notna()
                & (df["거리_m"] <= 700)
            ]
            add_record(
                records,
                seen_questions,
                f"{region}에서 {band}이면서 역 가까운 아파트 추천해줘",
                "multi_condition",
                pick_doc(subset, "거리_m", True),
                "거리_m",
                "RECOMMEND_STRUCTURED",
                "EXACT_MATCH" if not subset.empty else "NO_MATCH",
                must_not_recommend="Y" if subset.empty else "N",
            )
            infra_subset = df[
                (df["시군구"] == region)
                & (df["면적대"] == band)
                & df["병원_접근지표"].notna()
                & df["공원_접근지표"].notna()
            ]
            add_record(
                records,
                seen_questions,
                f"{region}에서 {band}이면서 공원과 병원 접근성 좋은 아파트 추천해줘",
                "multi_condition",
                pick_doc(infra_subset, "병원_접근지표", False),
                "병원_접근지표",
                "RECOMMEND_STRUCTURED",
                "EXACT_MATCH" if not infra_subset.empty else "UNKNOWN",
                must_not_recommend="Y" if infra_subset.empty else "N",
                must_disclose_limit="Y" if infra_subset.empty else "N",
            )

    return records[:target]


def build_region_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = top_values(df, "시군구", 20)
    top_dongs = top_values(df, "동", 20)
    top_sido = top_values(df, "시도", 10)

    for sido in top_sido:
        subset = df[df["시도"] == sido]
        add_record(
            records,
            seen_questions,
            f"{sido}에서 추천 가능한 아파트 알려줘",
            "region",
            pick_doc(subset, "세대수", False),
            "시도",
            "RECOMMEND_STRUCTURED",
            "EXACT_MATCH" if not subset.empty else "NO_MATCH",
            must_not_recommend="Y" if subset.empty else "N",
        )
    for region in top_regions:
        subset = df[df["시군구"] == region]
        add_record(
            records,
            seen_questions,
            f"{region} 아파트 뭐 있어",
            "region",
            pick_doc(subset, "세대수", False),
            "description",
            "GENERAL_RETRIEVAL_QA",
            "UNKNOWN",
            must_not_recommend="N",
        )
    for dong in top_dongs:
        subset = df[df["동"] == dong]
        add_record(
            records,
            seen_questions,
            f"{dong} 대표 단지 특징 설명해줘",
            "region",
            pick_doc(subset, "세대수", False),
            "description",
            "GENERAL_RETRIEVAL_QA",
            "UNKNOWN",
            must_not_recommend="N",
        )

    return records[:target]


def build_vague_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    prompts = [
        "괜찮은 아파트 추천해줘",
        "살기 좋은 단지 뭐 있어",
        "무난한 아파트 골라줘",
        "실거주 괜찮은 곳 추천해줘",
        "추천할 만한 단지 알려줘",
    ]
    for prompt in prompts:
        add_record(
            records,
            seen_questions,
            prompt,
            "vague",
            "",
            "지원 범위",
            "RECOMMEND_COMPARATIVE",
            "UNKNOWN",
            must_not_recommend="Y",
            must_disclose_limit="Y",
        )

    top_regions = top_values(df, "시군구", 12)
    for region in top_regions:
        add_record(
            records,
            seen_questions,
            f"{region}에서 괜찮은 아파트 추천해줘",
            "vague",
            "",
            "지원 범위",
            "RECOMMEND_COMPARATIVE",
            "UNKNOWN",
            must_not_recommend="Y",
            must_disclose_limit="Y",
        )

    return records[:target]


def build_colloquial_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    prompts = [
        "지하철 가까운곳 추천좀",
        "병원 가까운 데 있음?",
        "공원 많은 데 보여줘",
        "송파구 괜춘한 단지 뭐있어",
        "강동구 아파트 추천좀",
    ]
    for prompt in prompts:
        router = "RECOMMEND_COMPARATIVE" if "공원" in prompt or "병원" in prompt else "RECOMMEND_STRUCTURED"
        add_record(
            records,
            seen_questions,
            prompt,
            "colloquial",
            "",
            "질문 해석",
            router,
            "UNKNOWN",
            must_not_recommend="Y",
            must_disclose_limit="Y",
        )

    return records[:target]


def augment_questions(
    df: pd.DataFrame,
    qtype: str,
    target: int,
    records: list[dict[str, str]],
    seen_questions: set[str],
) -> list[dict[str, str]]:
    if len(records) >= target:
        return records[:target]

    top_regions = top_values(df, "시군구", 30)
    top_dongs = top_values(df, "동", 30)
    top_bands = top_values(df, "면적대", 6)
    if qtype == "condition":
        suffixes = [
            "아파트 추천해줘",
            "조건 맞는 단지 찾아줘",
            "추천 가능한 곳 보여줘",
            "후보 알려줘",
        ]
        for region in top_regions or ["서울"]:
            for band in top_bands or ["중형"]:
                for suffix in suffixes:
                    add_record(
                        records,
                        seen_questions,
                        f"{region}에서 {band} {suffix}",
                        qtype,
                        "",
                        "면적대",
                        "RECOMMEND_STRUCTURED",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]
    elif qtype == "comparison":
        themes = ["공원", "병원", "지하철"]
        endings = ["비교해줘", "비교 추천해줘", "좋은 순으로 보여줘", "후보 비교해줘"]
        for region in top_regions or ["서울"]:
            for theme in themes:
                for ending in endings:
                    add_record(
                        records,
                        seen_questions,
                        f"{region}에서 {theme} 기준 아파트 {ending}",
                        qtype,
                        "",
                        "비교 기준",
                        "RECOMMEND_COMPARATIVE",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]
    elif qtype == "multi_condition":
        addons = ["역 가까운", "공원 접근 좋은", "병원 접근 좋은", "가격 괜찮은"]
        endings = ["아파트 추천해줘", "단지 찾아줘", "후보 보여줘", "비교해줘"]
        for region in top_regions or ["서울"]:
            for band in top_bands or ["중형"]:
                for addon in addons:
                    for ending in endings:
                        add_record(
                            records,
                            seen_questions,
                            f"{region}에서 {band}이면서 {addon} {ending}",
                            qtype,
                            "",
                            "복합 조건",
                            "RECOMMEND_STRUCTURED",
                            "UNKNOWN",
                            "Y",
                            "Y",
                        )
                        if len(records) >= target:
                            return records[:target]
    elif qtype == "region":
        retrieval_endings = [
            "아파트 뭐 있어",
            "대표 단지 특징 설명해줘",
            "아파트 특징 요약해줘",
            "단지 분위기 정리해줘",
            "대표 아파트 알려줘",
            "주요 단지 알려줘",
            "단지 특징 알려줘",
            "지역 대표 단지 정리해줘",
            "지역 아파트 요약해줘",
            "아파트 분위기 알려줘",
            "주요 아파트 특징 설명해줘",
            "대표 단지 요약해줘",
            "아파트 정보 정리해줘",
            "대표 아파트 특징 설명해줘",
            "지역 단지 정보 알려줘",
        ]
        for region in top_regions or ["서울"]:
            for ending in retrieval_endings:
                add_record(
                    records,
                    seen_questions,
                    f"{region} {ending}",
                    qtype,
                    "",
                    "description",
                    "GENERAL_RETRIEVAL_QA",
                    "UNKNOWN",
                    "N",
                    "N",
                )
                if len(records) >= target:
                    return records[:target]
        for dong in top_dongs or ["지역"]:
            for ending in retrieval_endings:
                add_record(
                    records,
                    seen_questions,
                    f"{dong} {ending}",
                    qtype,
                    "",
                    "description",
                    "GENERAL_RETRIEVAL_QA",
                    "UNKNOWN",
                    "N",
                    "N",
                )
                if len(records) >= target:
                    return records[:target]
        prefixes = ["주변", "라인", "권역", "쪽"]
        endings = ["아파트 알려줘", "단지 뭐 있어", "추천해줘", "후보 보여줘"]
        for region in top_regions or ["서울"]:
            for prefix in prefixes:
                for ending in endings:
                    add_record(
                        records,
                        seen_questions,
                        f"{region} {prefix} {ending}",
                        qtype,
                        "",
                        "시군구",
                        "RECOMMEND_STRUCTURED",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]
        for dong in top_dongs or ["지역"]:
            for prefix in prefixes:
                for ending in endings:
                    add_record(
                        records,
                        seen_questions,
                        f"{dong} {prefix} {ending}",
                        qtype,
                        "",
                        "동",
                        "RECOMMEND_STRUCTURED",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]
    elif qtype == "vague":
        adjectives = ["괜찮은", "무난한", "살기 좋은", "추천할 만한", "편한", "실거주 좋은"]
        subjects = ["아파트", "단지", "곳", "후보"]
        endings = ["추천해줘", "찾아줘", "골라줘", "보여줘"]
        for region in top_regions or ["서울"]:
            for adjective in adjectives:
                for subject in subjects:
                    for ending in endings:
                        add_record(
                            records,
                            seen_questions,
                            f"{region}에서 {adjective} {subject} {ending}",
                            qtype,
                            "",
                            "지원 범위",
                            "RECOMMEND_COMPARATIVE",
                            "UNKNOWN",
                            "Y",
                            "Y",
                        )
                        if len(records) >= target:
                            return records[:target]
    else:
        snippets = ["추천좀", "뭐 있어", "찾아줘", "보여줘", "어디임", "괜춘함?", "있냐", "ㄱㄱ", "궁금", "봐줘"]
        themes = [
            "지하철 가까운곳",
            "병원 가까운곳",
            "공원 많은곳",
            "가성비 단지",
            "생활 편한곳",
            "규제 적은곳",
            "대단지 아파트",
        ]
        for theme in themes:
            for snippet in snippets:
                add_record(
                    records,
                    seen_questions,
                    f"{theme} {snippet}",
                    qtype,
                    "",
                    "질문 해석",
                    "RECOMMEND_STRUCTURED",
                    "UNKNOWN",
                    "Y",
                    "Y",
                )
                if len(records) >= target:
                    return records[:target]
        for region in top_regions or ["서울"]:
            for theme in themes:
                for snippet in snippets:
                    add_record(
                        records,
                        seen_questions,
                        f"{region} {theme} {snippet}",
                        qtype,
                        "",
                        "질문 해석",
                        "RECOMMEND_STRUCTURED",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]
        for dong in top_dongs or ["지역"]:
            for theme in ["아파트", "단지", "추천", "후보"]:
                for snippet in snippets:
                    add_record(
                        records,
                        seen_questions,
                        f"{dong} {theme} {snippet}",
                        qtype,
                        "",
                        "질문 해석",
                        "RECOMMEND_STRUCTURED",
                        "UNKNOWN",
                        "Y",
                        "Y",
                    )
                    if len(records) >= target:
                        return records[:target]

    return records[:target]


def main() -> None:
    random.seed(RANDOM_SEED)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    df, encoding = load_csv(INPUT_FILE)
    seen_questions: set[str] = set()
    all_records: list[dict[str, str]] = []

    generators = {
        "condition": build_condition_questions,
        "comparison": build_comparison_questions,
        "multi_condition": build_multi_condition_questions,
        "region": build_region_questions,
        "vague": build_vague_questions,
        "colloquial": build_colloquial_questions,
    }

    for qtype, target in TYPE_TARGETS.items():
        generated = generators[qtype](df, target, seen_questions)
        generated = augment_questions(df, qtype, target, generated, seen_questions)
        all_records.extend(generated[:target])

    if len(all_records) != TOTAL_TARGET:
        raise RuntimeError(f"총 질문 수가 {TOTAL_TARGET}개가 아닙니다: {len(all_records)}")

    all_df = pd.DataFrame(all_records)
    questions_df = all_df[["question", "type"]].copy()
    eval_df = all_df[
        [
            "question",
            "expected_doc",
            "expected_field",
            "expected_router_type",
            "expected_match_status",
            "must_not_recommend",
            "must_disclose_limit",
        ]
    ].copy()

    questions_df.to_csv(OUTPUT_QUESTIONS, index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUTPUT_EVAL, index=False, encoding="utf-8-sig")

    type_counts = questions_df["type"].value_counts().rename_axis("type").reset_index(name="count")
    sample_df = all_df.head(12)

    report_lines = [
        "# Edge Question Report",
        "",
        "## 개요",
        f"- 입력 파일: `{INPUT_FILE.name}`",
        f"- 입력 인코딩: `{encoding}`",
        f"- 총 생성 질문 수: {len(all_records)}",
        "",
        "## 유형별 개수",
        markdown_table(type_counts),
        "",
        "## 생성 규칙 요약",
        "- 추천형 질문에는 expected_router_type과 expected_match_status를 함께 기록했습니다.",
        "- no-match와 unsupported comparative 질문은 must_not_recommend=Y로 표시했습니다.",
        "- 한계 고지가 필요한 질문은 must_disclose_limit=Y로 표시했습니다.",
        "",
        "## 샘플 12개",
        markdown_table(sample_df),
        "",
        "## 생성 파일",
        f"- `{OUTPUT_QUESTIONS.name}`",
        f"- `{OUTPUT_EVAL.name}`",
        f"- `{OUTPUT_REPORT.name}`",
    ]
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
