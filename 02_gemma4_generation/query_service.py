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

UNSUPPORTED_COMPARATIVE_HINTS = ["아이 키우기", "살기 좋", "미래 가치", "투자 가치", "학군"]
META_HINTS = ["언제 기준", "기준 데이터", "무슨 질문", "어떤 질문", "답변 가능", "예시 질문"]
KNOWLEDGE_HINTS = ["뭐야", "설명", "뜻", "차이"]
RECOMMEND_HINTS = ["추천", "비교", "골라", "찾아", "뭐 있어", "보여줘"]


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def resolve_apartment_name(question: str, detailed_df: pd.DataFrame) -> str | None:
    if "아파트명" not in detailed_df.columns:
        return None
    candidates = [
        name
        for name in detailed_df["아파트명"].dropna().astype(str).unique().tolist()
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
    region = {"시도": None, "시군구": None, "동": None}
    for column in ["시도", "시군구", "동"]:
        if column not in detailed_df.columns:
            continue
        values = detailed_df[column].dropna().astype(str).unique().tolist()
        matches = [value for value in values if value and value in question]
        if matches:
            region[column] = max(matches, key=len)
    return region


def detect_comparative_tags(question: str) -> list[str]:
    return [tag for tag in SUPPORTED_COMPARATIVE_KEYWORDS if tag in question]


def parse_structured_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    price_match = re.search(r"(\d+)\s*억\s*이하", question)
    if price_match:
        filters["price_max_manwon"] = int(price_match.group(1)) * 10000

    distance_match = re.search(r"(\d+)\s*m", question, flags=re.IGNORECASE)
    if distance_match:
        filters["distance_subway_max_m"] = int(distance_match.group(1))

    area_match = re.search(r"전용\s*(\d+(?:\.\d+)?)", question)
    if area_match:
        filters["exclusive_area_min"] = float(area_match.group(1))

    if "신축" in question:
        filters["move_in_year_min"] = 2020
    if "대단지" in question:
        filters["households_min"] = 1000

    return filters


def route_query(question: str, detailed_df: pd.DataFrame, knowledge_df: pd.DataFrame) -> dict[str, Any]:
    question = normalize_question(question)
    apartment_name = resolve_apartment_name(question, detailed_df)
    knowledge_term = resolve_knowledge_term(question, knowledge_df)
    comparative_tags = detect_comparative_tags(question)
    region = resolve_region(question, detailed_df)
    filters = parse_structured_filters(question)
    wants_comparison = "비교" in question or "좋" in question

    if any(hint in question for hint in META_HINTS):
        query_type = "DATA_SCOPE_META"
    elif knowledge_term and any(hint in question for hint in KNOWLEDGE_HINTS):
        query_type = "REAL_ESTATE_KNOWLEDGE"
    elif apartment_name:
        query_type = "APARTMENT_FACT_LOOKUP"
    elif any(hint in question for hint in UNSUPPORTED_COMPARATIVE_HINTS):
        query_type = "RECOMMEND_COMPARATIVE"
    elif comparative_tags and wants_comparison and not filters:
        query_type = "RECOMMEND_COMPARATIVE"
    elif any(hint in question for hint in RECOMMEND_HINTS):
        query_type = "RECOMMEND_STRUCTURED"
    else:
        query_type = "GENERAL_RETRIEVAL_QA"

    return {
        "query_type": query_type,
        "target_apartment_name": apartment_name,
        "region": region,
        "filters": filters,
        "comparative_tags": comparative_tags,
        "knowledge_term": knowledge_term,
    }


def filter_by_region(df: pd.DataFrame, region: dict[str, str | None]) -> pd.DataFrame:
    filtered = df
    for column in ["시도", "시군구", "동"]:
        value = region.get(column)
        if value and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str) == value]
    return filtered


def apply_structured_filters(df: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    filtered = df
    unknown_reasons: list[str] = []

    if "price_max_manwon" in filters:
        if "공급액(만원)" in filtered.columns:
            filtered = filtered[filtered["공급액(만원)"].notna() & (filtered["공급액(만원)"] <= filters["price_max_manwon"])]
        else:
            unknown_reasons.append("가격")
    if "distance_subway_max_m" in filters:
        if "거리_m" in filtered.columns:
            filtered = filtered[filtered["거리_m"].notna() & (filtered["거리_m"] <= filters["distance_subway_max_m"])]
        else:
            unknown_reasons.append("지하철 거리")
    if "exclusive_area_min" in filters:
        if "전용면적" in filtered.columns:
            filtered = filtered[filtered["전용면적"].notna() & (filtered["전용면적"] >= filters["exclusive_area_min"])]
        else:
            unknown_reasons.append("전용면적")
    if "move_in_year_min" in filters:
        if "입주예정연도" in filtered.columns:
            filtered = filtered[filtered["입주예정연도"].notna() & (filtered["입주예정연도"] >= filters["move_in_year_min"])]
        else:
            unknown_reasons.append("입주예정연도")
    if "households_min" in filters:
        if "세대수" in filtered.columns:
            filtered = filtered[filtered["세대수"].notna() & (filtered["세대수"] >= filters["households_min"])]
        else:
            unknown_reasons.append("세대수")

    return filtered, unknown_reasons


def rank_structured_candidates(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "거리_m" in df.columns:
        sort_columns.append("거리_m")
        ascending.append(True)
    if "공급액(만원)" in df.columns:
        sort_columns.append("공급액(만원)")
        ascending.append(True)
    if "세대수" in df.columns:
        sort_columns.append("세대수")
        ascending.append(False)
    if not sort_columns:
        return df
    return df.sort_values(sort_columns, ascending=ascending, na_position="last")


def dedupe_by_apartment_name(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    if "아파트명" not in df.columns:
        return df.head(limit)
    return df.drop_duplicates(subset=["아파트명"], keep="first").head(limit)


def determine_data_cutoff(detailed_df: pd.DataFrame) -> str:
    if "데이터기준일" in detailed_df.columns and detailed_df["데이터기준일"].notna().any():
        return safe_text(detailed_df["데이터기준일"].dropna().iloc[0])
    return "기준일 미상"


def build_meta_answer(detailed_df: pd.DataFrame) -> str:
    cutoff = determine_data_cutoff(detailed_df)
    return (
        f"데이터 기준일: {cutoff}\n"
        "답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원 기반 비교\n"
        "예시 질문: 송파구에서 10억 이하 아파트 추천해줘 / 지하철 접근성 좋은 아파트 비교해줘 / 용적률이 뭐야"
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
    caution = safe_text(row.get("caution")) or ""
    answer = f"일반 설명: {definition}\n우리 데이터에서 대응 가능한 필드: {related_fields}"
    if caution:
        answer += f"\n주의사항: {caution}"
    answer += f"\n데이터 기준일: {determine_data_cutoff(detailed_df)}"
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
        used_fields=["데이터기준일", "답변가능범위"],
        detailed_df=detailed_df,
    )


def infer_lookup_fields(question: str) -> list[str]:
    mappings = [
        (["세대수"], ["세대수"]),
        (["용적률"], ["용적률"]),
        (["건폐율"], ["건폐율"]),
        (["분양가", "공급가"], ["공급액(만원)"]),
        (["가격", "시세"], ["가격요약", "공급액(만원)", "평당_공급액"]),
        (["평당"], ["평당_공급액"]),
        (["입주", "입주연도"], ["입주예정연도"]),
        (["역", "지하철"], ["가장가까운역", "거리_m", "가장가까운역_호선요약"]),
        (["병원", "의료"], ["의료시설_요약", "병원_비교요약", "병원_접근지표"]),
        (["생활 인프라", "상권", "생활"], ["생활인프라_요약"]),
        (["통근", "통학"], ["통근통학_요약"]),
        (["정책", "규제", "투기과열지구", "분양가 상한제"], ["정책특이사항_설명"]),
        (["건설사"], ["건설사_요약"]),
        (["공원"], ["공원_비교요약", "공원_접근지표"]),
    ]
    fields: list[str] = []
    for keywords, mapped in mappings:
        if any(keyword in question for keyword in keywords):
            fields.extend(mapped)
    return list(dict.fromkeys(fields or ["description"]))


def answer_apartment_fact(question: str, detailed_df: pd.DataFrame) -> dict[str, Any] | None:
    apartment_name = resolve_apartment_name(question, detailed_df)
    if not apartment_name:
        return None
    row = detailed_df[detailed_df["아파트명"].astype(str) == apartment_name].iloc[0]
    fields = infer_lookup_fields(question)
    snippets: list[str] = []
    for field in fields:
        value = safe_text(row.get(field))
        if value:
            snippets.append(f"{field}: {value}")
    if not snippets:
        snippets.append("해당 항목 데이터 없음")
    answer = f"{apartment_name} 정보입니다.\n" + "\n".join(snippets)
    answer += f"\n데이터 기준일: {determine_data_cutoff(detailed_df)}"
    doc_id = safe_text(row.get("문서ID"))
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


def answer_structured_recommendation(parsed: dict[str, Any], detailed_df: pd.DataFrame) -> dict[str, Any]:
    filtered = filter_by_region(detailed_df, parsed["region"])
    filtered, unknown_reasons = apply_structured_filters(filtered, parsed["filters"])

    if unknown_reasons:
        answer = (
            "현재 데이터로는 해당 조건을 판단할 수 없습니다.\n"
            f"판단 불가 항목: {', '.join(unknown_reasons)}\n"
            f"데이터 기준일: {determine_data_cutoff(detailed_df)}"
        )
        return build_result(
            answer=answer,
            answer_type="unknown_response",
            match_status="UNKNOWN",
            query_type=parsed["query_type"],
            used_fields=unknown_reasons,
            insufficient_context=True,
            detailed_df=detailed_df,
            limitations=unknown_reasons,
        )

    if filtered.empty:
        answer = (
            f"현재 데이터 기준({determine_data_cutoff(detailed_df)})으로 해당 조건에 맞는 아파트를 찾지 못했습니다.\n"
            "조건을 완화하거나 지역/예산 범위를 다시 지정해 주세요."
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

    ranked = dedupe_by_apartment_name(rank_structured_candidates(filtered), limit=3)
    cited_doc_ids = [safe_text(value) for value in ranked["문서ID"].tolist() if safe_text(value)]
    lines = ["조건에 맞는 아파트 후보입니다."]
    for _, row in ranked.iterrows():
        reason_bits = []
        if safe_text(row.get("가장가까운역")):
            reason_bits.append(f"역 {safe_text(row.get('가장가까운역'))}")
        if safe_text(row.get("거리_m")):
            reason_bits.append(f"{safe_text(row.get('거리_m'))}m")
        if safe_text(row.get("공급액(만원)")):
            reason_bits.append(f"공급액 {safe_text(row.get('공급액(만원)'))}만원")
        if safe_text(row.get("생활인프라_요약")):
            reason_bits.append(safe_text(row.get("생활인프라_요약")))
        lines.append(f"- {safe_text(row.get('아파트명'))}: " + ", ".join(reason_bits))
    lines.append(f"데이터 기준일: {determine_data_cutoff(detailed_df)}")
    lines.append("답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원 기반 비교")
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
    if any(hint in question for hint in UNSUPPORTED_COMPARATIVE_HINTS):
        answer = (
            "현재 MVP에서는 공원, 병원, 지하철처럼 데이터로 직접 판정 가능한 조건만 비교 추천할 수 있습니다.\n"
            "예시 질문: 공원 가까운 아파트 비교해줘 / 병원 접근성 좋은 곳 비교해줘 / 지하철 접근성 좋은 아파트 비교해줘\n"
            f"데이터 기준일: {determine_data_cutoff(detailed_df)}"
        )
        return build_result(
            answer=answer,
            answer_type="unsupported_comparative_response",
            match_status="UNKNOWN",
            query_type=parsed["query_type"],
            detailed_df=detailed_df,
            limitations=["지원하지 않는 추상 비교 질의"],
        )

    tags = parsed["comparative_tags"]
    if not tags:
        return build_result(
            answer="비교 기준을 해석하지 못했습니다. 공원, 병원, 지하철처럼 명시적인 조건으로 다시 질문해 주세요.",
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
            f"현재 데이터로는 {tag} 기준 비교를 판단할 수 없습니다.\n"
            f"데이터 기준일: {determine_data_cutoff(detailed_df)}"
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
    cited_doc_ids = [safe_text(value) for value in ranked["문서ID"].tolist() if safe_text(value)]
    lines = [f"비교 기준: {summary_column}"]
    for _, row in ranked.iterrows():
        lines.append(f"- {safe_text(row.get('아파트명'))}: {safe_text(row.get(summary_column))}")
    lines.append(f"데이터 기준일: {determine_data_cutoff(detailed_df)}")
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
