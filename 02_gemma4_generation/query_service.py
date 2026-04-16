from __future__ import annotations

import re
from typing import Any

import pandas as pd

from common import safe_text


SUPPORTED_COMPARATIVE_KEYWORDS = {
    "공원": ("공원_접근지표", "공원_비교요약"),
    "병원": ("병원_접근지표", "병원_비교요약"),
    "지하철": ("지하철_접근지표", "교통_비교요약"),
    "학교": ("학교_접근지표", "학교_비교요약"),
}
COMPARATIVE_TAG_ALIASES = {
    "공원": ["공원"],
    "병원": ["병원"],
    "지하철": ["지하철", "역"],
    "학교": ["학교", "초등학교"],
}
UNSUPPORTED_COMPARATIVE_HINTS = [
    "아이 키우기 좋은",
    "살기 좋은",
    "미래 가치",
    "투자 가치",
    "입지 좋은",
    "학군",
    "괜찮은",
    "무난한",
    "추천할 만한",
    "편한",
    "실거주 괜찮은",
    "실거주 좋은",
    "좋은 단지",
    "좋은 곳",
    "좋은 후보",
]
UNSUPPORTED_COMPARATIVE_SYNTAX_HINTS = ["기준", "좋은 순", "순으로", "후보 비교"]
AMBIGUOUS_STRUCTURED_SCOPE_HINTS = ["주변", "라인", "권역", "쪽"]
UNSUPPORTED_STRUCTURED_QUALIFIER_HINTS = [
    "가성비",
    "규제 적은곳",
    "규제 적은 곳",
    "대단지",
    "생활 편한곳",
    "생활 편한 곳",
    "괜춘한 단지",
    "괜찮은 단지",
    "좋은 단지",
]
VAGUE_STRUCTURED_AMENITY_HINTS = [
    "지하철 가까운곳",
    "지하철 가까운 곳",
    "병원 가까운곳",
    "병원 가까운 곳",
    "공원 많은곳",
    "공원 많은 곳",
]
VAGUE_STRUCTURED_SYNTAX_HINTS = [
    "추천좀",
    "추천해줘",
    "뭐 있어",
    "찾아줘",
    "보여줘",
    "어디임",
    "있냐",
    "ㄱㄱ",
    "궁금",
    "봐줘",
    "있음?",
    "괜춘함?",
]
VAGUE_COMPARATIVE_PLACE_HINTS = ["가까운 데", "많은 데"]
META_HINTS = ["데이터 기준", "가용 범위", "무슨 질문", "어떤 질문", "예시 질문"]
KNOWLEDGE_HINTS = ["무엇", "설명", "뜻", "차이"]
RECOMMEND_HINTS = ["추천", "찾아줘", "보여줘", "후보", "알려줘"]
COMPARISON_HINTS = ["비교", "더 좋은", "어디가 나아"]
HARD_SELECTION_HINTS = ["추천", "찾아줘", "보여줘", "후보", "골라줘"]
GENERAL_RETRIEVAL_HINTS = [
    "뭐 있어",
    "어떤 아파트가 있어",
    "어떤 단지가 있어",
    "특징 설명",
    "설명해줘",
    "요약해줘",
    "정리해줘",
    "어떤 편이야",
    "어떤 편인지",
    "대표 아파트",
    "주요 단지",
    "단지 특징",
    "대표 단지",
    "아파트 분위기",
    "단지 정보",
    "아파트 정보",
]
SELECTION_HINTS = ["추천", "찾아줘", "보여줘", "후보", "알려줘", "골라줘"]
SUBJECTIVE_COMPARATIVE_HINTS = [
    "괜찮은",
    "무난한",
    "추천할 만한",
    "편한",
    "실거주 괜찮은",
    "실거주 좋은",
    "살기 좋은",
]
SUPPORTED_STRUCTURED_HINTS = [
    "가격 괜찮은",
    "가격 괜춘한",
    "역 가까운",
    "지하철 가까운 아파트",
    "공원과 병원 접근성 좋은",
]
AREA_BAND_RANGES = {
    "소형": (0.0, 60.0),
    "중소형": (60.0, 85.0),
    "중형": (85.0, 102.0),
    "중대형": (102.0, 135.0),
    "대형": (135.0, None),
}

APARTMENT_NAME_COL = "아파트명"
DOC_ID_COL = "문서ID"
REGION_COLUMNS = ("시도", "시군구", "동")
EXCLUSIVE_AREA_COL = "전용면적"
PRICE_COL = "공급액(만원)"
HOUSEHOLDS_COL = "세대수"
MOVE_IN_YEAR_COL = "입주예정연도"
SUBWAY_NAME_COL = "가장가까운역"
SUBWAY_DISTANCE_COL = "거리_m"
SUBWAY_SUMMARY_COL = "가장가까운역_호선요약"
LIFESTYLE_SUMMARY_COL = "생활인프라_요약"
MEDICAL_SUMMARY_COL = "의료시설_요약"
COMMUTE_SUMMARY_COL = "통근통학_요약"
POLICY_SUMMARY_COL = "정책특이사항_설명"
DESCRIPTION_COL = "description"
DATA_CUTOFF_COL = "데이터기준일"
AREA_BAND_COL = "면적대"


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def resolve_apartment_name(question: str, detailed_df: pd.DataFrame) -> str | None:
    if APARTMENT_NAME_COL not in detailed_df.columns:
        return None
    candidates = [
        name
        for name in detailed_df[APARTMENT_NAME_COL].dropna().astype(str).unique().tolist()
        if name and name in question
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def resolve_knowledge_term(question: str, knowledge_df: pd.DataFrame) -> str | None:
    if "term" not in knowledge_df.columns:
        return None
    candidates = [
        term
        for term in knowledge_df["term"].dropna().astype(str).unique().tolist()
        if term and term in question
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def resolve_region(question: str, detailed_df: pd.DataFrame) -> dict[str, str | None]:
    region = {column: None for column in REGION_COLUMNS}
    for column in REGION_COLUMNS:
        if column not in detailed_df.columns:
            continue
        values = detailed_df[column].dropna().astype(str).unique().tolist()
        matches = [value for value in values if value and value in question]
        if matches:
            region[column] = max(matches, key=len)
    return region


def detect_comparative_tags(question: str) -> list[str]:
    tags: list[str] = []
    for tag, aliases in COMPARATIVE_TAG_ALIASES.items():
        matched = False
        for alias in aliases:
            if alias == "역":
                if re.search(r"(?<!지)(?<!구)(?<!권)역", question):
                    matched = True
                    break
                continue
            if alias in question:
                matched = True
                break
        if matched:
            tags.append(tag)
    return tags


def detect_area_band(question: str) -> str | None:
    for band in AREA_BAND_RANGES:
        if band in question:
            return band
    return None


def is_unsupported_comparative_request(question: str, comparative_tags: list[str]) -> bool:
    if "가격 괜찮은" in question or "가격 괜춘한" in question:
        return False
    if any(hint in question for hint in UNSUPPORTED_COMPARATIVE_HINTS):
        return True
    if comparative_tags and any(hint in question for hint in UNSUPPORTED_COMPARATIVE_SYNTAX_HINTS):
        return True
    return False


def is_vague_comparative_place_request(question: str, comparative_tags: list[str]) -> bool:
    return bool(comparative_tags) and any(hint in question for hint in VAGUE_COMPARATIVE_PLACE_HINTS)


def is_subjective_comparative_request(question: str) -> bool:
    if "가격 괜찮은" in question or "가격 괜춘한" in question:
        return False
    has_subjective_hint = any(hint in question for hint in SUBJECTIVE_COMPARATIVE_HINTS)
    has_selection_hint = any(hint in question for hint in SELECTION_HINTS)
    return has_subjective_hint and has_selection_hint


def is_general_retrieval_request(
    question: str,
    region: dict[str, str | None],
    filters: dict[str, Any],
    comparative_tags: list[str],
    apartment_name: str | None,
    knowledge_term: str | None,
) -> bool:
    if apartment_name or knowledge_term or comparative_tags:
        return False
    if any(hint in question for hint in AMBIGUOUS_STRUCTURED_SCOPE_HINTS):
        return False
    if any(hint in question for hint in UNSUPPORTED_STRUCTURED_QUALIFIER_HINTS):
        return False
    if any(hint in question for hint in VAGUE_STRUCTURED_AMENITY_HINTS):
        return False
    if has_explicit_structured_filter(filters):
        return False
    if any(hint in question for hint in HARD_SELECTION_HINTS):
        return False
    if any(hint in question for hint in COMPARISON_HINTS):
        return False
    has_region = any(region.values())
    has_retrieval_hint = any(hint in question for hint in GENERAL_RETRIEVAL_HINTS)
    return has_region and has_retrieval_hint


def has_explicit_structured_filter(filters: dict[str, Any]) -> bool:
    explicit_keys = {
        "price_max_manwon",
        "distance_subway_max_m",
        "exclusive_area_min",
        "exclusive_area_max",
        "move_in_year_min",
        "households_min",
        "prefer_low_price",
        "prefer_near_subway",
        "prefer_park_hospital_access",
    }
    return any(key in filters for key in explicit_keys)


def is_ambiguous_structured_scope(question: str, region: dict[str, str | None], filters: dict[str, Any]) -> bool:
    has_region = any(region.values())
    has_ambiguous_scope = any(hint in question for hint in AMBIGUOUS_STRUCTURED_SCOPE_HINTS)
    return has_region and has_ambiguous_scope and not has_explicit_structured_filter(filters)


def is_unsupported_structured_request(
    question: str,
    region: dict[str, str | None],
    filters: dict[str, Any],
    comparative_tags: list[str],
) -> bool:
    has_region = any(region.values())
    area_band = safe_text(filters.get("area_band_label"))
    has_explicit_filter = has_explicit_structured_filter(filters)
    has_objective_filter = any(
        key in filters
        for key in {
            "price_max_manwon",
            "distance_subway_max_m",
            "move_in_year_min",
            "households_min",
            "prefer_low_price",
            "prefer_near_subway",
            "prefer_park_hospital_access",
        }
    )
    wants_recommendation = any(hint in question for hint in RECOMMEND_HINTS)
    has_vague_syntax = any(hint in question for hint in VAGUE_STRUCTURED_SYNTAX_HINTS)
    has_supported_structured_phrase = any(hint in question for hint in SUPPORTED_STRUCTURED_HINTS)

    if any(hint in question for hint in UNSUPPORTED_STRUCTURED_QUALIFIER_HINTS) and not has_objective_filter:
        return True
    if any(hint in question for hint in VAGUE_STRUCTURED_AMENITY_HINTS):
        return True
    if area_band and ("가격 괜찮은" in question or "가격 괜춘한" in question):
        return True
    if area_band and not has_objective_filter and not has_supported_structured_phrase:
        return True
    if has_region and wants_recommendation and not has_explicit_filter and not comparative_tags and not has_supported_structured_phrase:
        return True
    if has_region and has_vague_syntax and not has_explicit_filter and not comparative_tags and not has_supported_structured_phrase:
        return True
    return False


def parse_structured_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    price_match = re.search(r"(\d+)\s*억\s*이하", question)
    if price_match:
        filters["price_max_manwon"] = int(price_match.group(1)) * 10000

    distance_match = re.search(r"(\d+)\s*m\s*이내", question, flags=re.IGNORECASE)
    if distance_match:
        filters["distance_subway_max_m"] = int(distance_match.group(1))

    area_match = re.search(r"전용(?:면적)?\s*(\d+(?:\.\d+)?)", question)
    if area_match:
        filters["exclusive_area_min"] = float(area_match.group(1))

    households_match = re.search(r"(\d+)\s*세대\s*이상", question)
    if households_match:
        filters["households_min"] = int(households_match.group(1))

    if "신축" in question:
        filters["move_in_year_min"] = 2020
    if "가격 괜찮은" in question or "가격 괜춘한" in question:
        filters["prefer_low_price"] = True
    if "역 가까운" in question or "지하철 가까운 아파트" in question:
        filters["prefer_near_subway"] = True
        # Treat implicit "near subway" requests as a bounded distance filter so
        # obviously far candidates do not slip into exact-match recommendations.
        filters.setdefault("distance_subway_max_m", 500)
    if "공원과 병원 접근성 좋은" in question:
        filters["prefer_park_hospital_access"] = True

    area_band = detect_area_band(question)
    if area_band:
        filters["area_band_label"] = area_band

    return filters


def route_query(question: str, detailed_df: pd.DataFrame, knowledge_df: pd.DataFrame) -> dict[str, Any]:
    question = normalize_question(question)
    apartment_name = resolve_apartment_name(question, detailed_df)
    knowledge_term = resolve_knowledge_term(question, knowledge_df)
    comparative_tags = detect_comparative_tags(question)
    region = resolve_region(question, detailed_df)
    filters = parse_structured_filters(question)

    wants_recommendation = any(hint in question for hint in RECOMMEND_HINTS)
    wants_comparison = any(hint in question for hint in COMPARISON_HINTS)
    unsupported_comparative = is_unsupported_comparative_request(question, comparative_tags)
    vague_comparative_place_request = is_vague_comparative_place_request(question, comparative_tags)
    subjective_comparative_request = is_subjective_comparative_request(question)
    general_retrieval_request = is_general_retrieval_request(
        question,
        region,
        filters,
        comparative_tags,
        apartment_name,
        knowledge_term,
    )
    ambiguous_structured_scope = is_ambiguous_structured_scope(question, region, filters)
    unsupported_structured_request = is_unsupported_structured_request(question, region, filters, comparative_tags)

    if any(hint in question for hint in META_HINTS):
        query_type = "DATA_SCOPE_META"
    elif knowledge_term and any(hint in question for hint in KNOWLEDGE_HINTS):
        query_type = "REAL_ESTATE_KNOWLEDGE"
    elif apartment_name:
        query_type = "APARTMENT_FACT_LOOKUP"
    elif general_retrieval_request:
        query_type = "GENERAL_RETRIEVAL_QA"
    elif subjective_comparative_request:
        query_type = "RECOMMEND_COMPARATIVE"
    elif ambiguous_structured_scope or unsupported_structured_request:
        query_type = "RECOMMEND_STRUCTURED"
    elif vague_comparative_place_request:
        query_type = "RECOMMEND_COMPARATIVE"
    elif unsupported_comparative:
        query_type = "RECOMMEND_COMPARATIVE"
    elif comparative_tags and wants_comparison:
        query_type = "RECOMMEND_COMPARATIVE"
    elif wants_recommendation or filters or region["시도"] or region["시군구"] or region["동"]:
        query_type = "RECOMMEND_STRUCTURED"
    else:
        query_type = "GENERAL_RETRIEVAL_QA"

    return {
        "query_type": query_type,
        "target_apartment_name": apartment_name,
        "region": region,
        "filters": filters,
        "comparative_tags": comparative_tags,
        "unsupported_comparative": unsupported_comparative,
        "vague_comparative_place_request": vague_comparative_place_request,
        "ambiguous_structured_scope": ambiguous_structured_scope,
        "unsupported_structured_request": unsupported_structured_request,
        "knowledge_term": knowledge_term,
    }


def filter_by_region(df: pd.DataFrame, region: dict[str, str | None]) -> pd.DataFrame:
    filtered = df
    for column in REGION_COLUMNS:
        value = region.get(column)
        if value and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == value]
    return filtered


def apply_structured_filters(df: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    filtered = df
    unknown_reasons: list[str] = []

    if "area_band_label" in filters:
        if AREA_BAND_COL in filtered.columns:
            filtered = filtered[filtered[AREA_BAND_COL].astype(str) == safe_text(filters["area_band_label"])]
        else:
            unknown_reasons.append("???")

    if "price_max_manwon" in filters:
        if PRICE_COL in filtered.columns:
            filtered = filtered[filtered[PRICE_COL].notna() & (filtered[PRICE_COL] <= filters["price_max_manwon"])]
        else:
            unknown_reasons.append("가격")
    if "distance_subway_max_m" in filters:
        if SUBWAY_DISTANCE_COL in filtered.columns:
            filtered = filtered[filtered[SUBWAY_DISTANCE_COL].notna() & (filtered[SUBWAY_DISTANCE_COL] <= filters["distance_subway_max_m"])]
        else:
            unknown_reasons.append("지하철 거리")
    if "exclusive_area_min" in filters:
        if EXCLUSIVE_AREA_COL in filtered.columns:
            filtered = filtered[filtered[EXCLUSIVE_AREA_COL].notna() & (filtered[EXCLUSIVE_AREA_COL] >= filters["exclusive_area_min"])]
        else:
            unknown_reasons.append("전용면적")
    if "exclusive_area_max" in filters:
        if EXCLUSIVE_AREA_COL in filtered.columns:
            filtered = filtered[filtered[EXCLUSIVE_AREA_COL].notna() & (filtered[EXCLUSIVE_AREA_COL] < filters["exclusive_area_max"])]
        else:
            unknown_reasons.append("전용면적")
    if "move_in_year_min" in filters:
        if MOVE_IN_YEAR_COL in filtered.columns:
            filtered = filtered[filtered[MOVE_IN_YEAR_COL].notna() & (filtered[MOVE_IN_YEAR_COL] >= filters["move_in_year_min"])]
        else:
            unknown_reasons.append("입주예정연도")
    if "households_min" in filters:
        if HOUSEHOLDS_COL in filtered.columns:
            filtered = filtered[filtered[HOUSEHOLDS_COL].notna() & (filtered[HOUSEHOLDS_COL] >= filters["households_min"])]
        else:
            unknown_reasons.append("세대수")

    return filtered, unknown_reasons


def rank_structured_candidates(df: pd.DataFrame, filters: dict[str, Any] | None = None) -> pd.DataFrame:
    filters = filters or {}
    sort_columns: list[str] = []
    ascending: list[bool] = []

    if filters.get("prefer_park_hospital_access"):
        if "공원_접근지표" in df.columns:
            sort_columns.append("공원_접근지표")
            ascending.append(False)
        if "병원_접근지표" in df.columns:
            sort_columns.append("병원_접근지표")
            ascending.append(False)
    if filters.get("prefer_near_subway") and SUBWAY_DISTANCE_COL in df.columns:
        sort_columns.append(SUBWAY_DISTANCE_COL)
        ascending.append(True)
    if filters.get("prefer_low_price") and PRICE_COL in df.columns:
        sort_columns.append(PRICE_COL)
        ascending.append(True)

    if not sort_columns:
        if SUBWAY_DISTANCE_COL in df.columns:
            sort_columns.append(SUBWAY_DISTANCE_COL)
            ascending.append(True)
        if PRICE_COL in df.columns:
            sort_columns.append(PRICE_COL)
            ascending.append(True)
        if HOUSEHOLDS_COL in df.columns:
            sort_columns.append(HOUSEHOLDS_COL)
            ascending.append(False)

    if not sort_columns:
        return df
    return df.sort_values(sort_columns, ascending=ascending, na_position="last")


def dedupe_by_apartment_name(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    if APARTMENT_NAME_COL not in df.columns:
        return df.head(limit)
    return df.drop_duplicates(subset=[APARTMENT_NAME_COL], keep="first").head(limit)


def determine_data_cutoff(detailed_df: pd.DataFrame) -> str:
    if DATA_CUTOFF_COL in detailed_df.columns and detailed_df[DATA_CUTOFF_COL].notna().any():
        return safe_text(detailed_df[DATA_CUTOFF_COL].dropna().iloc[0])
    return "기준일 미상"


def build_meta_answer(detailed_df: pd.DataFrame) -> str:
    cutoff = determine_data_cutoff(detailed_df)
    return (
        f"데이터 기준: {cutoff}\n"
        "답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원/학교 기반 추천과 비교\n"
        "예시 질문: 송파구에서 10억 이하 아파트 추천해줘 / 병원 접근 좋은 아파트 추천해줘 / 학교 접근 좋은 단지 비교해줘"
    )


def build_result(
    *,
    answer: str,
    answer_type: str,
    match_status: str,
    query_type: str,
    cited_doc_ids: list[str] | None = None,
    top_doc_id: str = "",
    used_fields: list[str] | None = None,
    insufficient_context: bool = False,
    detailed_df: pd.DataFrame,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "answer_type": answer_type,
        "match_status": match_status,
        "query_type": query_type,
        "cited_doc_ids": cited_doc_ids or [],
        "top_doc_id": top_doc_id,
        "used_fields": used_fields or [],
        "data_cutoff": determine_data_cutoff(detailed_df),
        "limitations": limitations or [],
        "insufficient_context": insufficient_context,
    }


def answer_knowledge(question: str, knowledge_df: pd.DataFrame, detailed_df: pd.DataFrame) -> dict[str, Any] | None:
    term = resolve_knowledge_term(question, knowledge_df)
    if not term:
        return None
    row = knowledge_df[knowledge_df["term"].astype(str) == term].iloc[0]
    definition = safe_text(row.get("definition"))
    related_fields = safe_text(row.get("related_dataset_fields")) or "없음"
    caution = safe_text(row.get("caution"))
    answer = f"일반 설명: {definition}\n우리 데이터에 연결 가능한 필드: {related_fields}"
    if caution:
        answer += f"\n주의사항: {caution}"
    answer += f"\n데이터 기준: {determine_data_cutoff(detailed_df)}"
    return build_result(
        answer=answer,
        answer_type="knowledge_answer",
        match_status="KNOWN",
        query_type="REAL_ESTATE_KNOWLEDGE",
        used_fields=related_fields.split("|") if related_fields else [],
        detailed_df=detailed_df,
    )


def answer_meta(detailed_df: pd.DataFrame) -> dict[str, Any]:
    return build_result(
        answer=build_meta_answer(detailed_df),
        answer_type="meta_answer",
        match_status="KNOWN",
        query_type="DATA_SCOPE_META",
        used_fields=[DATA_CUTOFF_COL, "답변가능범위"],
        detailed_df=detailed_df,
    )


def infer_lookup_fields(question: str) -> list[str]:
    mappings = [
        (["세대수"], [HOUSEHOLDS_COL]),
        (["전용면적"], [EXCLUSIVE_AREA_COL]),
        (["공급면적"], ["공급면적"]),
        (["분양가", "공급가"], [PRICE_COL]),
        (["가격", "시세"], ["가격요약", PRICE_COL, "평당_공급액"]),
        (["평당"], ["평당_공급액"]),
        (["입주", "입주연도"], [MOVE_IN_YEAR_COL]),
        (["역", "지하철"], [SUBWAY_NAME_COL, SUBWAY_DISTANCE_COL, SUBWAY_SUMMARY_COL]),
        (["병원", "의료"], [MEDICAL_SUMMARY_COL, "병원_비교요약", "병원_접근지표"]),
        (["생활 인프라", "상권", "생활"], [LIFESTYLE_SUMMARY_COL]),
        (["통근", "통학"], [COMMUTE_SUMMARY_COL]),
        (["정책", "규제", "투기과열지구", "분양가 상한제"], [POLICY_SUMMARY_COL]),
        (["건설사"], ["건설사_요약"]),
        (["공원"], ["공원_비교요약", "공원_접근지표"]),
    ]
    fields: list[str] = []
    for keywords, mapped in mappings:
        if any(keyword in question for keyword in keywords):
            fields.extend(mapped)
    return list(dict.fromkeys(fields or [DESCRIPTION_COL]))


def answer_apartment_fact(question: str, detailed_df: pd.DataFrame) -> dict[str, Any] | None:
    apartment_name = resolve_apartment_name(question, detailed_df)
    if not apartment_name:
        return None
    row = detailed_df[detailed_df[APARTMENT_NAME_COL].astype(str) == apartment_name].iloc[0]
    fields = infer_lookup_fields(question)
    snippets: list[str] = []
    for field in fields:
        value = safe_text(row.get(field))
        if value:
            snippets.append(f"{field}: {value}")
    if not snippets:
        snippets.append("해당 항목 데이터가 없습니다")
    answer = f"{apartment_name} 정보입니다.\n" + "\n".join(snippets)
    answer += f"\n데이터 기준: {determine_data_cutoff(detailed_df)}"
    doc_id = safe_text(row.get(DOC_ID_COL))
    return build_result(
        answer=answer,
        answer_type="apartment_fact_lookup",
        match_status="EXACT_MATCH",
        query_type="APARTMENT_FACT_LOOKUP",
        cited_doc_ids=[doc_id] if doc_id else [],
        top_doc_id=doc_id,
        used_fields=fields,
        detailed_df=detailed_df,
    )


def build_disclosure_lines(detailed_df: pd.DataFrame) -> list[str]:
    return [
        f"데이터 기준: {determine_data_cutoff(detailed_df)}",
        "답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원/학교 기반 비교",
    ]


def build_unknown_response(reason_fields: list[str], query_type: str, detailed_df: pd.DataFrame) -> dict[str, Any]:
    answer = (
        "현재 데이터로는 해당 조건을 안전하게 추천 기준으로 확정할 수 없습니다.\n"
        f"판단 보류 항목: {', '.join(reason_fields)}\n"
        + "\n".join(build_disclosure_lines(detailed_df))
    )
    return build_result(
        answer=answer,
        answer_type="unknown_response",
        match_status="UNKNOWN",
        query_type=query_type,
        used_fields=reason_fields,
        insufficient_context=True,
        detailed_df=detailed_df,
        limitations=reason_fields,
    )


def answer_structured_recommendation(parsed: dict[str, Any], detailed_df: pd.DataFrame) -> dict[str, Any]:
    if parsed.get("ambiguous_structured_scope"):
        return build_unknown_response(["범위 표현"], parsed["query_type"], detailed_df)
    if parsed.get("unsupported_structured_request"):
        return build_unknown_response(["지원하지 않는 추천 표현"], parsed["query_type"], detailed_df)

    filtered = filter_by_region(detailed_df, parsed["region"])
    filtered, unknown_reasons = apply_structured_filters(filtered, parsed["filters"])

    if unknown_reasons:
        return build_unknown_response(unknown_reasons, parsed["query_type"], detailed_df)

    if filtered.empty:
        answer = (
            f"현재 데이터 기준({determine_data_cutoff(detailed_df)})으로 해당 조건에 맞는 아파트를 찾지 못했습니다.\n"
            "조건을 완화하거나 지역, 예산, 거리 기준을 다시 지정해 주세요.\n"
            + "\n".join(build_disclosure_lines(detailed_df))
        )
        return build_result(
            answer=answer,
            answer_type="no_match_response",
            match_status="NO_MATCH",
            query_type=parsed["query_type"],
            used_fields=list(parsed["filters"].keys()),
            detailed_df=detailed_df,
            limitations=["조건 일치 후보 없음"],
        )

    ranked = dedupe_by_apartment_name(rank_structured_candidates(filtered, parsed["filters"]), limit=3)
    cited_doc_ids = [safe_text(value) for value in ranked[DOC_ID_COL].tolist() if safe_text(value)]
    lines = ["조건에 맞는 아파트 후보입니다."]
    for _, row in ranked.iterrows():
        reason_bits: list[str] = []
        station = safe_text(row.get(SUBWAY_NAME_COL))
        if station:
            reason_bits.append(f"역 {station}")
        distance = safe_text(row.get(SUBWAY_DISTANCE_COL))
        if distance:
            reason_bits.append(f"{distance}m")
        price = safe_text(row.get(PRICE_COL))
        if price:
            reason_bits.append(f"공급액 {price}만원")
        area = safe_text(row.get(EXCLUSIVE_AREA_COL))
        if area:
            reason_bits.append(f"전용 {area}㎡")
        lifestyle = safe_text(row.get(LIFESTYLE_SUMMARY_COL))
        if lifestyle:
            reason_bits.append(lifestyle)
        lines.append(f"- {safe_text(row.get(APARTMENT_NAME_COL))}: " + ", ".join(reason_bits))
    lines.extend(build_disclosure_lines(detailed_df))
    return build_result(
        answer="\n".join(lines),
        answer_type="recommendation",
        match_status="EXACT_MATCH",
        query_type=parsed["query_type"],
        cited_doc_ids=cited_doc_ids,
        top_doc_id=cited_doc_ids[0] if cited_doc_ids else "",
        used_fields=list(parsed["filters"].keys()) or ["시군구"],
        detailed_df=detailed_df,
    )


def answer_comparative_recommendation(question: str, parsed: dict[str, Any], detailed_df: pd.DataFrame) -> dict[str, Any]:
    if parsed.get("unsupported_comparative") or parsed.get("vague_comparative_place_request"):
        answer = (
            "현재 MVP에서는 공원, 병원, 지하철, 학교처럼 데이터로 직접 비교 가능한 기준만 지원합니다.\n"
            "예시 질문: 공원 접근 좋은 아파트 비교해줘 / 병원 접근 좋은 단지 비교해줘 / 지하철 접근 좋은 단지 비교해줘\n"
            + "\n".join(build_disclosure_lines(detailed_df))
        )
        return build_result(
            answer=answer,
            answer_type="unsupported_comparative_response",
            match_status="UNKNOWN",
            query_type=parsed["query_type"],
            detailed_df=detailed_df,
            limitations=["지원하지 않는 비교 질의"],
        )

    tags = parsed["comparative_tags"]
    if not tags:
        return build_result(
            answer="비교 기준을 해석하지 못했습니다. 공원, 병원, 지하철, 학교처럼 명시적인 기준으로 다시 질문해 주세요.",
            answer_type="unsupported_comparative_response",
            match_status="UNKNOWN",
            query_type=parsed["query_type"],
            detailed_df=detailed_df,
            limitations=["비교 기준 해석 실패"],
            insufficient_context=True,
        )

    tag = tags[0]
    score_column, summary_column = SUPPORTED_COMPARATIVE_KEYWORDS[tag]
    filtered = filter_by_region(detailed_df, parsed["region"])
    if score_column not in filtered.columns:
        filtered = filtered.iloc[0:0]
    else:
        filtered = filtered[filtered[score_column].notna()]

    if filtered.empty:
        answer = (
            f"현재 데이터로는 {tag} 기준 비교를 안전하게 판단할 수 없습니다.\n"
            + "\n".join(build_disclosure_lines(detailed_df))
        )
        return build_result(
            answer=answer,
            answer_type="unknown_response",
            match_status="UNKNOWN",
            query_type=parsed["query_type"],
            used_fields=[score_column],
            detailed_df=detailed_df,
            limitations=[score_column],
            insufficient_context=True,
        )

    ranked = dedupe_by_apartment_name(filtered.sort_values(score_column, ascending=False), limit=3)
    cited_doc_ids = [safe_text(value) for value in ranked[DOC_ID_COL].tolist() if safe_text(value)]
    lines = [f"비교 기준: {summary_column}"]
    for _, row in ranked.iterrows():
        lines.append(f"- {safe_text(row.get(APARTMENT_NAME_COL))}: {safe_text(row.get(summary_column))}")
    lines.extend(build_disclosure_lines(detailed_df))
    return build_result(
        answer="\n".join(lines),
        answer_type="comparison_recommendation",
        match_status="EXACT_MATCH",
        query_type=parsed["query_type"],
        cited_doc_ids=cited_doc_ids,
        top_doc_id=cited_doc_ids[0] if cited_doc_ids else "",
        used_fields=[score_column, summary_column],
        detailed_df=detailed_df,
    )


def answer_query(question: str, detailed_df: pd.DataFrame, knowledge_df: pd.DataFrame) -> dict[str, Any] | None:
    parsed = route_query(question, detailed_df, knowledge_df)
    if parsed["query_type"] == "DATA_SCOPE_META":
        return answer_meta(detailed_df)
    if parsed["query_type"] == "REAL_ESTATE_KNOWLEDGE":
        return answer_knowledge(question, knowledge_df, detailed_df)
    if parsed["query_type"] == "APARTMENT_FACT_LOOKUP":
        return answer_apartment_fact(question, detailed_df)
    if parsed["query_type"] == "RECOMMEND_STRUCTURED":
        return answer_structured_recommendation(parsed, detailed_df)
    if parsed["query_type"] == "RECOMMEND_COMPARATIVE":
        return answer_comparative_recommendation(question, parsed, detailed_df)
    return None
