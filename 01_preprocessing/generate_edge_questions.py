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
    "condition": 334,
    "comparison": 334,
    "multi_condition": 333,
    "region": 333,
    "vague": 333,
    "colloquial": 333,
}
ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV를 읽지 못했습니다: {last_error}")


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
        }
    )
    return True


def pick_doc(df: pd.DataFrame, sort_by: str, ascending: bool = True) -> str | None:
    if sort_by not in df.columns or df.empty:
        return None
    picked = df.sort_values(sort_by, ascending=ascending).iloc[0]
    return q(picked.get("문서ID"))


def make_condition_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    distances = [300, 500, 700, 1000]
    area_bands = [band for band in df["면적대"].dropna().astype(str).unique().tolist() if band]
    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()[:12]
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()[:12]

    for dist in distances:
        subset = df[df["거리_m"].notna() & (df["거리_m"] <= dist)]
        add_record(records, seen_questions, f"지하철 {dist}m 이내 아파트 알려줘", "condition", pick_doc(subset, "거리_m", True), "거리_m")
        add_record(records, seen_questions, f"역까지 {dist}m 안쪽 아파트 뭐 있어", "condition", pick_doc(subset, "거리_m", True), "거리_m")

    for band in area_bands:
        subset = df[df["면적대"] == band]
        add_record(records, seen_questions, f"{band} 아파트 알려줘", "condition", pick_doc(subset, "전용면적", True), "면적대")
        add_record(records, seen_questions, f"{band} 면적대 단지 뭐 있어", "condition", pick_doc(subset, "전용면적", True), "면적대")

    medical_subset = df[df["의료시설_요약"].fillna("").str.contains("상급|의료기관|병원", regex=True)]
    infra_subset = df[df["생활인프라_요약"].fillna("").str.contains("풍부|무난|편의시설", regex=True)]
    major_builder_subset = df[df["건설사_요약"].fillna("").str.contains("대형 건설사")]
    policy_subset = df[df["정책특이사항_설명"].fillna("").str.contains("투기과열지구|분양가상한제")]
    low_price_subset = df[df["가격요약"].fillna("").str.contains("합리적")]
    high_price_subset = df[df["가격요약"].fillna("").str.contains("높은 편")]

    fixed_prompts = [
        ("병원 가까운 아파트 뭐 있어", medical_subset, "의료시설_요약"),
        ("상급 의료기관 접근 가능한 단지 알려줘", medical_subset, "의료시설_요약"),
        ("생활 인프라 좋은 아파트 뭐 있어", infra_subset, "생활인프라_요약"),
        ("상권 잘 갖춰진 아파트 알려줘", infra_subset, "생활인프라_요약"),
        ("대형 건설사 아파트 뭐 있어", major_builder_subset, "건설사_요약"),
        ("규제 적용된 아파트 알려줘", policy_subset, "정책특이사항_설명"),
        ("분양가상한제 적용 단지 뭐 있어", policy_subset, "정책특이사항_설명"),
        ("가격 합리적인 아파트 알려줘", low_price_subset, "가격요약"),
        ("평당 가격 높은 아파트 뭐 있어", high_price_subset, "가격요약"),
    ]
    for prompt, subset, field in fixed_prompts:
        add_record(records, seen_questions, prompt, "condition", pick_doc(subset, field if field in subset.columns else "문서ID", True), field)

    for region in top_regions:
        subset = df[df["시군구"] == region]
        add_record(records, seen_questions, f"{region}에서 지하철 가까운 아파트 알려줘", "condition", pick_doc(subset[subset["거리_m"].notna()], "거리_m", True), "거리_m")
        add_record(records, seen_questions, f"{region}에서 세대수 많은 단지 있어", "condition", pick_doc(subset, "세대수", False), "세대수")

    for dong in top_dongs:
        subset = df[df["동"] == dong]
        add_record(records, seen_questions, f"{dong}에서 생활 편한 아파트 알려줘", "condition", pick_doc(subset, "세대수", False), "생활인프라_요약")
        add_record(records, seen_questions, f"{dong}에서 병원 접근 좋은 단지 뭐 있어", "condition", pick_doc(subset, "세대수", False), "의료시설_요약")

    return records[:target]


def make_comparison_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()[:15]
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()[:15]

    global_prompts = [
        ("세대수 가장 많은 아파트는", df, "세대수", False),
        ("세대수 가장 적은 아파트는", df, "세대수", True),
        ("평당 가격 가장 높은 아파트는", df, "평당_공급액", False),
        ("평당 가격 가장 낮은 아파트는", df, "평당_공급액", True),
        ("지하철 가장 가까운 아파트는", df[df["거리_m"].notna()], "거리_m", True),
        ("지하철 가장 먼 아파트는", df[df["거리_m"].notna()], "거리_m", False),
        ("공급액이 가장 높은 아파트는", df, "공급액(만원)", False),
        ("공급액이 가장 낮은 아파트는", df, "공급액(만원)", True),
        ("전용면적 가장 큰 아파트는", df, "전용면적", False),
        ("전용면적 가장 작은 아파트는", df, "전용면적", True),
    ]
    for prompt, subset, field, ascending in global_prompts:
        add_record(records, seen_questions, prompt, "comparison", pick_doc(subset, field, ascending), field)

    for region in top_regions:
        region_df = df[df["시군구"] == region]
        prompts = [
            (f"{region}에서 세대수 가장 많은 아파트는", "세대수", False),
            (f"{region}에서 평당 가격 가장 높은 아파트는", "평당_공급액", False),
            (f"{region}에서 지하철 제일 가까운 아파트는", "거리_m", True),
            (f"{region}에서 공급액 가장 낮은 단지는", "공급액(만원)", True),
            (f"{region}에서 전용면적 가장 큰 아파트는", "전용면적", False),
        ]
        for prompt, field, ascending in prompts:
            subset = region_df if field != "거리_m" else region_df[region_df["거리_m"].notna()]
            add_record(records, seen_questions, prompt, "comparison", pick_doc(subset, field, ascending), field)

    for dong in top_dongs:
        dong_df = df[df["동"] == dong]
        prompts = [
            (f"{dong}에서 세대수 가장 큰 단지는", "세대수", False),
            (f"{dong}에서 평당 가격 가장 비싼 아파트는", "평당_공급액", False),
            (f"{dong}에서 역이 제일 가까운 아파트는", "거리_m", True),
            (f"{dong}에서 공급액 낮은 아파트는", "공급액(만원)", True),
        ]
        for prompt, field, ascending in prompts:
            subset = dong_df if field != "거리_m" else dong_df[dong_df["거리_m"].notna()]
            add_record(records, seen_questions, prompt, "comparison", pick_doc(subset, field, ascending), field)

    return records[:target]


def make_multi_condition_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()[:14]
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()[:12]
    bands = [band for band in df["면적대"].dropna().astype(str).unique().tolist() if band]

    condition_sets = [
        ("지하철 가까우면서 가격 낮은 아파트", (df["거리_m"].notna()) & (df["거리_m"] <= 700) & (df["가격요약"].fillna("").str.contains("합리적")), "가격요약", "거리_m"),
        ("역세권이면서 세대수 많은 아파트", (df["거리_m"].notna()) & (df["거리_m"] <= 500), "세대수", None),
        ("대형 건설사이면서 생활 인프라 좋은 아파트", df["건설사_요약"].fillna("").str.contains("대형 건설사") & df["생활인프라_요약"].fillna("").str.contains("풍부|무난"), "생활인프라_요약", None),
        ("규제 적용되면서 지하철 가까운 아파트", df["정책특이사항_설명"].fillna("").str.contains("투기과열지구|분양가상한제") & (df["거리_m"].notna()) & (df["거리_m"] <= 700), "정책특이사항_설명", None),
        ("병원 접근 좋고 생활 편한 아파트", df["의료시설_요약"].fillna("").str.contains("병원|의료기관") & df["생활인프라_요약"].fillna("").str.contains("풍부|무난"), "의료시설_요약", None),
    ]
    for prompt, mask, field, sort_field in condition_sets:
        subset = df[mask]
        expected_doc = pick_doc(subset, sort_field, True) if sort_field else q(subset.iloc[0]["문서ID"]) if not subset.empty else None
        add_record(records, seen_questions, prompt, "multi_condition", expected_doc, field)

    for region in top_regions:
        region_df = df[df["시군구"] == region]
        prompts = [
            (f"{region}에서 세대수 많으면서 지하철 가까운 아파트", region_df[(region_df["거리_m"].notna()) & (region_df["거리_m"] <= 700)], "세대수"),
            (f"{region}에서 가격 낮고 생활 편한 아파트", region_df[region_df["가격요약"].fillna("").str.contains("합리적")], "가격요약"),
            (f"{region}에서 규제 있으면서 대형 건설사인 단지", region_df[region_df["정책특이사항_설명"].fillna("").str.contains("투기과열지구|분양가상한제") & region_df["건설사_요약"].fillna("").str.contains("대형 건설사")], "정책특이사항_설명"),
            (f"{region}에서 병원 가깝고 역 가까운 아파트", region_df[region_df["거리_m"].notna() & (region_df["거리_m"] <= 700) & region_df["의료시설_요약"].fillna("").str.contains("병원|의료기관")], "의료시설_요약"),
        ]
        for prompt, subset, field in prompts:
            expected_doc = pick_doc(subset, "거리_m", True) if "거리_m" in subset.columns and subset["거리_m"].notna().any() else q(subset.iloc[0]["문서ID"]) if not subset.empty else None
            add_record(records, seen_questions, prompt, "multi_condition", expected_doc, field)

    for dong in top_dongs:
        dong_df = df[df["동"] == dong]
        prompts = [
            (f"{dong}에서 가격 괜찮고 세대수 큰 아파트", dong_df[dong_df["가격요약"].fillna("").str.contains("합리적")], "가격요약"),
            (f"{dong}에서 중형이면서 역 가까운 단지", dong_df[(dong_df["면적대"] == "중형") & dong_df["거리_m"].notna() & (dong_df["거리_m"] <= 700)], "면적대"),
            (f"{dong}에서 생활 인프라 좋고 병원 접근 괜찮은 아파트", dong_df[dong_df["생활인프라_요약"].fillna("").str.contains("풍부|무난") & dong_df["의료시설_요약"].fillna("").str.contains("병원|의료기관")], "생활인프라_요약"),
        ]
        for prompt, subset, field in prompts:
            expected_doc = pick_doc(subset, "거리_m", True) if "거리_m" in subset.columns and subset["거리_m"].notna().any() else q(subset.iloc[0]["문서ID"]) if not subset.empty else None
            add_record(records, seen_questions, prompt, "multi_condition", expected_doc, field)

    for band in bands:
        subset = df[(df["면적대"] == band) & df["거리_m"].notna() & (df["거리_m"] <= 700)]
        add_record(records, seen_questions, f"{band}이면서 지하철 가까운 아파트 알려줘", "multi_condition", pick_doc(subset, "거리_m", True), "면적대")
        subset2 = df[(df["면적대"] == band) & df["건설사_요약"].fillna("").str.contains("대형 건설사")]
        add_record(records, seen_questions, f"{band} 중에서 대형 건설사 단지 뭐 있어", "multi_condition", q(subset2.iloc[0]["문서ID"]) if not subset2.empty else None, "건설사_요약")

    return records[:target]


def make_region_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()[:30]
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()[:30]
    top_sido = df["시도"].dropna().astype(str).value_counts().index.tolist()[:10]

    for region in top_regions:
        subset = df[df["시군구"] == region]
        prompts = [
            f"{region} 아파트 뭐 있어",
            f"{region} 아파트 알려줘",
            f"{region}에 분양 단지 있어",
            f"{region} 지역 아파트 리스트 보여줘",
        ]
        for prompt in prompts:
            add_record(records, seen_questions, prompt, "region", q(subset.iloc[0]["문서ID"]) if not subset.empty else None, "시군구")

    for dong in top_dongs:
        subset = df[df["동"] == dong]
        prompts = [
            f"{dong} 아파트 알려줘",
            f"{dong}에 있는 아파트 뭐 있어",
            f"{dong} 단지 찾아줘",
            f"{dong} 쪽 청약 아파트 있어",
        ]
        for prompt in prompts:
            add_record(records, seen_questions, prompt, "region", q(subset.iloc[0]["문서ID"]) if not subset.empty else None, "동")

    for sido in top_sido:
        subset = df[df["시도"] == sido]
        prompts = [
            f"{sido} 아파트 알려줘",
            f"{sido} 분양 아파트 뭐 있어",
            f"{sido}에서 청약 가능한 단지 있어",
        ]
        for prompt in prompts:
            add_record(records, seen_questions, prompt, "region", q(subset.iloc[0]["문서ID"]) if not subset.empty else None, "시도")

    return records[:target]


def make_vague_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_region = q(df["시군구"].dropna().astype(str).value_counts().index[0])
    top_dong = q(df["동"].dropna().astype(str).value_counts().index[0])
    best_living = df.sort_values(["세대수", "거리_m"], ascending=[False, True]).iloc[0]
    best_transport = df[df["거리_m"].notna()].sort_values("거리_m", ascending=True).iloc[0]
    best_value = df[df["가격요약"].fillna("").str.contains("합리적")].iloc[0] if not df[df["가격요약"].fillna("").str.contains("합리적")].empty else df.iloc[0]

    base_prompts = [
        ("살기 좋은 아파트 어디야", q(best_living["문서ID"]), "description"),
        ("생활 편한 아파트 추천해줘", q(best_living["문서ID"]), "생활인프라_요약"),
        ("교통 괜찮은 아파트 뭐가 좋아", q(best_transport["문서ID"]), "거리_m"),
        ("가성비 괜찮은 단지 추천해줘", q(best_value["문서ID"]), "가격요약"),
        ("가족이 살기 무난한 아파트 어디야", q(best_living["문서ID"]), "구조요약"),
        ("역세권 느낌 나는 아파트 추천해줘", q(best_transport["문서ID"]), "거리_m"),
        ("병원 이용 편한 아파트 있을까", q(best_living["문서ID"]), "의료시설_요약"),
        ("생활하기 덜 불편한 단지 뭐 있어", q(best_living["문서ID"]), "생활인프라_요약"),
        (f"{top_region}에서 살기 괜찮은 아파트 어디야", q(best_living["문서ID"]), "시군구"),
        (f"{top_dong} 쪽에서 살기 좋은 단지 추천해줘", q(best_living["문서ID"]), "동"),
    ]
    for prompt, doc, field in base_prompts:
        add_record(records, seen_questions, prompt, "vague", doc, field)

    adjective_sets = [
        "무난한", "괜찮은", "살기 편한", "눈여겨볼 만한", "실거주하기 좋은", "생활 편한",
        "교통 괜찮은", "가성비 괜찮은", "규제 부담 적은", "주변 편의시설 괜찮은",
    ]
    noun_sets = ["아파트", "단지", "곳", "매물 말고 분양 단지", "청약 단지"]
    doc_cycle = [q(best_living["문서ID"]), q(best_transport["문서ID"]), q(best_value["문서ID"])]

    idx = 0
    for adjective in adjective_sets:
        for noun in noun_sets:
            prompt = f"{adjective} {noun} 추천해줘"
            add_record(records, seen_questions, prompt, "vague", doc_cycle[idx % len(doc_cycle)], "description")
            idx += 1
            prompt2 = f"{adjective} {noun} 어디 있어"
            add_record(records, seen_questions, prompt2, "vague", doc_cycle[idx % len(doc_cycle)], "description")
            idx += 1

    return records[:target]


def make_colloquial_questions(df: pd.DataFrame, target: int, seen_questions: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()[:25]
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()[:20]
    top_transport_doc = q(df[df["거리_m"].notna()].sort_values("거리_m").iloc[0]["문서ID"])
    value_doc = q(df.iloc[0]["문서ID"])

    base_prompts = [
        ("지하철 가까운곳 추천좀", top_transport_doc, "거리_m"),
        ("병원 가까운 데 있음?", value_doc, "의료시설_요약"),
        ("생활 편한곳 어디임", value_doc, "생활인프라_요약"),
        ("가성비 괜찮은 아파트 머있어", value_doc, "가격요약"),
        ("세대수 큰 단지 뭐냐", value_doc, "세대수"),
        ("규제 걸린 아파트 있음?", value_doc, "정책특이사항_설명"),
        ("역세권 비슷한곳 추천해줘", top_transport_doc, "거리_m"),
        ("살기 괜찮은데 어딘지 좀", value_doc, "description"),
    ]
    for prompt, doc, field in base_prompts:
        add_record(records, seen_questions, prompt, "colloquial", doc, field)

    for region in top_regions:
        subset = df[df["시군구"] == region]
        doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else ""
        prompts = [
            f"{region} 아파트 머있어",
            f"{region} 쪽 단지 뭐 있음",
            f"{region}에서 괜찮은 곳 추천좀",
            f"{region} 역 가까운 데 있음?",
            f"{region}에 큰 단지 있냐",
        ]
        for prompt in prompts:
            field = "시군구" if "아파트" in prompt or "단지" in prompt else "description"
            add_record(records, seen_questions, prompt, "colloquial", doc, field)

    for dong in top_dongs:
        subset = df[df["동"] == dong]
        doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else ""
        prompts = [
            f"{dong} 아파트 알려주라",
            f"{dong} 쪽 괜찮냐",
            f"{dong} 근처 지하철 편함?",
            f"{dong} 생활 인프라 어때보임",
            f"{dong} 쪽 추천좀",
        ]
        for prompt in prompts:
            add_record(records, seen_questions, prompt, "colloquial", doc, "동")

    fillers = [
        "추천좀", "머있어", "어디임", "괜춘함?", "좋은편임?", "어때", "있냐", "어디쪽이냐",
    ]
    themes = [
        ("지하철 가까운", top_transport_doc, "거리_m"),
        ("생활 편한", value_doc, "생활인프라_요약"),
        ("가격 괜찮은", value_doc, "가격요약"),
        ("병원 가까운", value_doc, "의료시설_요약"),
        ("규제 적은", value_doc, "정책특이사항_설명"),
    ]
    for theme, doc, field in themes:
        for filler in fillers:
            add_record(records, seen_questions, f"{theme}곳 {filler}", "colloquial", doc, field)

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

    top_regions = df["시군구"].dropna().astype(str).value_counts().index.tolist()
    top_dongs = df["동"].dropna().astype(str).value_counts().index.tolist()
    area_bands = [band for band in df["면적대"].dropna().astype(str).unique().tolist() if band]

    if qtype == "condition":
        for _, row in df.sort_values(["세대수", "평당_공급액"], ascending=[False, True]).iterrows():
            region = q(row.get("시군구"))
            dong = q(row.get("동"))
            band = q(row.get("면적대"))
            doc = q(row.get("문서ID"))
            station = q(row.get("가장가까운역"))
            if region:
                add_record(records, seen_questions, f"{region}에서 {band} 아파트 있어", qtype, doc, "면적대")
                add_record(records, seen_questions, f"{region}에서 규제 있는 아파트 알려줘", qtype, doc, "정책특이사항_설명")
            if dong:
                add_record(records, seen_questions, f"{dong}에서 생활 편한 아파트 뭐 있어", qtype, doc, "생활인프라_요약")
                add_record(records, seen_questions, f"{dong}에서 병원 접근 좋은 아파트 알려줘", qtype, doc, "의료시설_요약")
            if station:
                add_record(records, seen_questions, f"{station} 가까운 아파트 뭐 있어", qtype, doc, "가장가까운역")
            if len(records) >= target:
                break

    elif qtype == "comparison":
        for region in top_regions:
            region_df = df[df["시군구"] == region]
            prompts = [
                (f"{region}에서 병원 접근 제일 좋은 아파트는", "의료시설_요약"),
                (f"{region}에서 생활 인프라 가장 좋은 단지는", "생활인프라_요약"),
                (f"{region}에서 규제 강한 아파트는", "정책특이사항_설명"),
                (f"{region}에서 면적 제일 넓은 아파트는", "전용면적"),
                (f"{region}에서 가격 메리트 큰 단지는", "가격요약"),
            ]
            for prompt, field in prompts:
                if field == "전용면적":
                    doc = pick_doc(region_df, field, False)
                else:
                    doc = q(region_df.iloc[0]["문서ID"]) if not region_df.empty else None
                add_record(records, seen_questions, prompt, qtype, doc, field)
                if len(records) >= target:
                    break
            if len(records) >= target:
                break

    elif qtype == "multi_condition":
        for region in top_regions:
            for band in area_bands:
                subset = df[(df["시군구"] == region) & (df["면적대"] == band)]
                doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else None
                add_record(records, seen_questions, f"{region}에서 {band}이면서 생활 편한 아파트", qtype, doc, "생활인프라_요약")
                add_record(records, seen_questions, f"{region}에서 {band}이면서 가격 괜찮은 단지", qtype, doc, "가격요약")
                add_record(records, seen_questions, f"{region}에서 {band}이면서 역 가까운 아파트", qtype, doc, "거리_m")
                if len(records) >= target:
                    break
            if len(records) >= target:
                break

        for dong in top_dongs:
            subset = df[df["동"] == dong]
            doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else None
            add_record(records, seen_questions, f"{dong}에서 규제 있으면서 교통 좋은 단지", qtype, doc, "정책특이사항_설명")
            add_record(records, seen_questions, f"{dong}에서 병원 가깝고 생활 편한 아파트", qtype, doc, "의료시설_요약")
            if len(records) >= target:
                break

    elif qtype == "region":
        for region in top_regions:
            subset = df[df["시군구"] == region]
            doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else None
            add_record(records, seen_questions, f"{region} 쪽 아파트 뭐 있지", qtype, doc, "시군구")
            add_record(records, seen_questions, f"{region} 근처 분양 단지 찾아줘", qtype, doc, "시군구")
            if len(records) >= target:
                break

        for dong in top_dongs:
            subset = df[df["동"] == dong]
            doc = q(subset.iloc[0]["문서ID"]) if not subset.empty else None
            add_record(records, seen_questions, f"{dong} 라인 아파트 뭐 있어", qtype, doc, "동")
            add_record(records, seen_questions, f"{dong} 주변 단지 알려줘", qtype, doc, "동")
            if len(records) >= target:
                break

    elif qtype == "vague":
        fallback_doc = q(df.iloc[0]["문서ID"])
        adjectives = ["무난한", "괜찮은", "추천할 만한", "실거주 좋은", "편한", "덜 불편한", "살만한", "눈여겨볼 만한", "고민해볼 만한"]
        subjects = ["아파트", "단지", "곳", "데", "청약 단지", "분양 단지", "집"]
        endings = ["추천해줘", "어디야", "뭐 있어", "골라줘", "알려줘", "있을까", "궁금해", "찾아줘"]
        for adjective in adjectives:
            for subject in subjects:
                for ending in endings:
                    add_record(records, seen_questions, f"{adjective} {subject} {ending}", qtype, fallback_doc, "description")
                    if len(records) >= target:
                        break
                if len(records) >= target:
                    break
            if len(records) >= target:
                break

        region_prompts = [
            "어디가 살기 편해", "어디가 무난해", "어디가 덜 불편해", "어디가 실거주 괜찮아", "어디가 생활 편해",
            "어디가 교통 괜찮아", "어디가 가성비 괜찮아", "어디가 추천할 만해",
        ]
        for region in top_regions[:12]:
            for prompt in region_prompts:
                add_record(records, seen_questions, f"{region}에서 {prompt}", qtype, fallback_doc, "description")
                if len(records) >= target:
                    break
            if len(records) >= target:
                break

    elif qtype == "colloquial":
        fallback_doc = q(df.iloc[0]["문서ID"])
        snippets = [
            "머있어", "추천좀", "어디임", "괜춘함?", "어딘데", "어때보임", "있냐", "좀 봐줘",
            "좋냐", "가볼만함?", "찾아줘", "보여줘",
        ]
        themes = ["강동구 아파트", "지하철 가까운곳", "생활 편한곳", "병원 가까운데", "가성비 단지", "규제 적은곳"]
        for theme in themes:
            for snippet in snippets:
                add_record(records, seen_questions, f"{theme} {snippet}", qtype, fallback_doc, "description")
                if len(records) >= target:
                    break
            if len(records) >= target:
                break

    return records[:target]


def main() -> None:
    random.seed(RANDOM_SEED)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    df, encoding = load_csv(INPUT_FILE)
    if "문서ID" not in df.columns:
        raise RuntimeError("문서ID 컬럼이 필요합니다.")

    seen_questions: set[str] = set()
    all_records: list[dict[str, str]] = []

    generators = {
        "condition": make_condition_questions,
        "comparison": make_comparison_questions,
        "multi_condition": make_multi_condition_questions,
        "region": make_region_questions,
        "vague": make_vague_questions,
        "colloquial": make_colloquial_questions,
    }

    for qtype, target in TYPE_TARGETS.items():
        generated = generators[qtype](df, target, seen_questions)
        generated = augment_questions(df, qtype, target, generated, seen_questions)
        if len(generated) < 300:
            raise RuntimeError(f"{qtype} 유형 질문이 300개 미만입니다: {len(generated)}")
        all_records.extend(generated)

    if len(all_records) < TOTAL_TARGET:
        fallback_doc = q(df.iloc[0]["문서ID"])
        fillers = [
            ("생활 편한곳 추천좀 더", "colloquial", "생활인프라_요약"),
            ("지하철 가까운 단지 더 알려줘", "condition", "거리_m"),
            ("가성비 괜찮은 곳 더 있어", "vague", "가격요약"),
            ("강동구 아파트 머있어 더", "colloquial", "시군구"),
            ("역세권이면서 큰 단지 또 뭐 있어", "multi_condition", "세대수"),
            ("평당 가격 제일 높은 아파트 또 알려줘", "comparison", "평당_공급액"),
            ("둔촌동 아파트 뭐 있어 더", "region", "동"),
            ("살기 무난한 단지 하나 더 추천해줘", "vague", "description"),
        ]
        for question, qtype, field in fillers:
            if len(all_records) >= TOTAL_TARGET:
                break
            if add_record(all_records, seen_questions, question, qtype, fallback_doc, field):
                continue

    if len(all_records) != TOTAL_TARGET:
        raise RuntimeError(f"총 질문 수가 {TOTAL_TARGET}개가 아닙니다: {len(all_records)}")

    questions_df = pd.DataFrame(all_records)[["question", "type"]]
    eval_df = pd.DataFrame(all_records)[["question", "expected_doc", "expected_field"]]

    questions_df.to_csv(OUTPUT_QUESTIONS, index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUTPUT_EVAL, index=False, encoding="utf-8-sig")

    type_counts = questions_df["type"].value_counts().rename_axis("type").reset_index(name="count")
    sample_df = pd.DataFrame(all_records).head(12)

    report_lines = [
        "# Edge Question Report",
        "",
        "## 개요",
        f"- 입력 파일: `{INPUT_FILE.name}`",
        f"- 감지 인코딩: `{encoding}`",
        f"- 총 생성 질문 수: {len(all_records)}",
        "",
        "## 유형별 개수",
        markdown_table(type_counts),
        "",
        "## 생성 규칙 요약",
        "- 조건 질문: 거리, 가격, 병원, 정책, 건설사, 면적대 조건을 반영했습니다.",
        "- 비교 질문: 세대수, 가격, 면적, 역 접근성 기준 최댓값/최솟값 질문을 만들었습니다.",
        "- 복합 조건 질문: 지역 + 교통 + 가격 + 정책 + 인프라 조합을 사용했습니다.",
        "- 지역 질문: 시도, 시군구, 동 단위 질문을 생성했습니다.",
        "- 모호 질문: 추천/살기 좋음/생활 편의 같은 추상 표현을 사용했습니다.",
        "- 구어체 질문: 오타/축약/구어형 표현을 반영했습니다.",
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
