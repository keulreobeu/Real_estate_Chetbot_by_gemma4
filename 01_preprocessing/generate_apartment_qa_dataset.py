from __future__ import annotations

import json
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
INPUT_KNOWLEDGE = QA_DIR / "real_estate_knowledge_base.csv"
OUTPUT_QA = QA_DIR / "apartment_qa_dataset.csv"
OUTPUT_JSONL = QA_DIR / "apartment_finetune_dataset.jsonl"
OUTPUT_EVAL = QA_DIR / "evaluation_dataset.csv"
OUTPUT_REPORT = REPORT_DIR / "03_qa_generation_report.md"

RANDOM_SEED = 42
MIN_QA_TARGET = 50000
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


def safe_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


def fmt_int(value: Any) -> str | None:
    if is_missing(value):
        return None
    return f"{int(round(float(value))):,}"


def fmt_float(value: Any, digits: int = 1) -> str | None:
    if is_missing(value):
        return None
    return f"{float(value):,.{digits}f}"


def fmt_area(value: Any) -> str | None:
    area = fmt_float(value, 2)
    return f"{area}㎡" if area else None


def build_location(row: pd.Series) -> str | None:
    parts = [safe_text(row.get("시도")), safe_text(row.get("시군구")), safe_text(row.get("동"))]
    parts = [part for part in parts if part]
    return " ".join(parts) if parts else None


def build_subject(row: pd.Series, duplicated_names: set[str]) -> str:
    name = safe_text(row.get("아파트명")) or "이 아파트"
    if name not in duplicated_names:
        return name

    exclusive_area = fmt_float(row.get("전용면적"), 2)
    supply_area = fmt_float(row.get("공급면적"), 2)
    unit_type = safe_text(row.get("타입"))
    parts = [name]
    if exclusive_area:
        parts.append(f"전용 {exclusive_area}㎡")
    if supply_area:
        parts.append(f"공급 {supply_area}㎡")
    if unit_type:
        parts.append(f"타입 {unit_type}")
    return " ".join(parts)


def make_answer_fact_area(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    area = fmt_area(row.get("전용면적"))
    if not name or not area:
        return None
    return f"{name}의 전용면적은 {area}입니다."


def make_answer_supply_area(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    area = fmt_area(row.get("공급면적"))
    if not name or not area:
        return None
    return f"{name}의 공급면적은 {area}입니다."


def make_answer_households(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    households = fmt_int(row.get("세대수"))
    if not name or not households:
        return None
    return f"{name}은 총 {households}세대 규모입니다."


def make_answer_structure(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    structure = safe_text(row.get("구조요약"))
    if not name or not structure:
        return None
    return f"{name}의 구조 요약은 {structure}"


def make_answer_area_band(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    band = safe_text(row.get("면적대"))
    if not name or not band:
        return None
    return f"{name}은 {band} 면적대로 분류됩니다."


def make_answer_location(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    location = build_location(row)
    if not name or not location:
        return None
    return f"{name}은 {location}에 위치합니다."


def make_answer_location_detail(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    address = safe_text(row.get("법정동주소"))
    if not name or not address:
        return None
    return f"{name}의 주소는 {address}입니다."


def make_answer_station(row: pd.Series) -> str | None:
    station = safe_text(row.get("가장가까운역"))
    distance = fmt_int(row.get("거리_m"))
    if not station:
        return None
    if distance:
        return f"가장 가까운 역은 {station}이며 약 {distance}m 거리입니다."
    return f"가장 가까운 역은 {station}입니다."


def make_answer_transport_summary(row: pd.Series) -> str | None:
    summary = safe_text(row.get("교통_비교요약")) or safe_text(row.get("가장가까운역_호선요약"))
    return summary


def make_answer_distance(row: pd.Series) -> str | None:
    distance = fmt_int(row.get("거리_m"))
    if not distance:
        return None
    return f"가장 가까운 역까지 거리는 약 {distance}m입니다."


def make_answer_price(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    price = fmt_int(row.get("공급액(만원)"))
    if not name or not price:
        return None
    return f"{name}의 공급액은 {price}만원입니다."


def make_answer_price_per_pyeong(row: pd.Series) -> str | None:
    value = fmt_float(row.get("평당_공급액"), 2)
    if not value:
        return None
    return f"평당 공급액은 {value}만원입니다."


def make_answer_price_summary(row: pd.Series) -> str | None:
    return safe_text(row.get("가격요약"))


def make_answer_medical(row: pd.Series) -> str | None:
    return safe_text(row.get("의료시설_요약"))


def make_answer_lifestyle(row: pd.Series) -> str | None:
    return safe_text(row.get("생활인프라_요약"))


def make_answer_commute(row: pd.Series) -> str | None:
    return safe_text(row.get("통근통학_요약"))


def make_answer_builder(row: pd.Series) -> str | None:
    return safe_text(row.get("건설사_요약"))


def make_answer_policy(row: pd.Series) -> str | None:
    return safe_text(row.get("정책특이사항_설명"))


def make_answer_description(row: pd.Series) -> str | None:
    return safe_text(row.get("description"))


def build_templates(subject: str) -> list[dict[str, str]]:
    return [
        {"category": "fact", "question": f"{subject} 전용면적은 얼마야", "answer_key": "fact_area"},
        {"category": "fact", "question": f"{subject} 공급면적은 얼마야", "answer_key": "supply_area"},
        {"category": "fact", "question": f"{subject} 세대수는 몇 세대야", "answer_key": "households"},
        {"category": "fact", "question": f"{subject} 구조는 어떻게 돼", "answer_key": "structure"},
        {"category": "fact", "question": f"{subject} 면적대는 뭐야", "answer_key": "area_band"},
        {"category": "location", "question": f"{subject} 위치가 어디야", "answer_key": "location"},
        {"category": "location", "question": f"{subject} 주소 알려줘", "answer_key": "location_detail"},
        {"category": "transport", "question": f"{subject} 근처 지하철역은 어디야", "answer_key": "station"},
        {"category": "transport", "question": f"{subject} 지하철 접근성 어때", "answer_key": "transport_summary"},
        {"category": "transport", "question": f"{subject} 가까운 역까지 거리는 얼마야", "answer_key": "distance"},
        {"category": "price", "question": f"{subject} 분양가는 얼마야", "answer_key": "price"},
        {"category": "price", "question": f"{subject} 평당 가격은", "answer_key": "price_per_pyeong"},
        {"category": "price", "question": f"{subject} 가격 수준은 어떤 편이야", "answer_key": "price_summary"},
        {"category": "lifestyle", "question": f"{subject} 생활 인프라는 어떤 편이야", "answer_key": "lifestyle"},
        {"category": "lifestyle", "question": f"{subject} 병원 접근성은 어때", "answer_key": "medical"},
        {"category": "lifestyle", "question": f"{subject} 통근 통학 여건은 어때", "answer_key": "commute"},
        {"category": "lifestyle", "question": f"{subject} 건설사 수준은 어때", "answer_key": "builder"},
        {"category": "policy", "question": f"{subject} 규제지역이야", "answer_key": "policy"},
        {"category": "fact", "question": f"{subject} 단지 설명해줘", "answer_key": "description"},
    ]


ANSWER_BUILDERS = {
    "fact_area": make_answer_fact_area,
    "supply_area": make_answer_supply_area,
    "households": make_answer_households,
    "structure": make_answer_structure,
    "area_band": make_answer_area_band,
    "location": make_answer_location,
    "location_detail": make_answer_location_detail,
    "station": make_answer_station,
    "transport_summary": make_answer_transport_summary,
    "distance": make_answer_distance,
    "price": make_answer_price,
    "price_per_pyeong": make_answer_price_per_pyeong,
    "price_summary": make_answer_price_summary,
    "medical": make_answer_medical,
    "lifestyle": make_answer_lifestyle,
    "commute": make_answer_commute,
    "builder": make_answer_builder,
    "policy": make_answer_policy,
    "description": make_answer_description,
}


def make_knowledge_records(knowledge_df: pd.DataFrame) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for index, row in knowledge_df.iterrows():
        term = safe_text(row.get("term"))
        definition = safe_text(row.get("definition"))
        related_fields = safe_text(row.get("related_dataset_fields")) or "없음"
        caution = safe_text(row.get("caution")) or ""
        if not term or not definition:
            continue
        answer = f"일반 설명: {definition}\n우리 데이터에서 대응 가능한 필드: {related_fields}"
        if caution:
            answer += f"\n주의사항: {caution}"
        base = {
            "아파트명": "GENERAL_KNOWLEDGE",
            "문서ID": f"KNOW_{index + 1:03d}",
            "category": "knowledge_term",
            "answer_type": "knowledge_answer",
            "expected_answer_type": "knowledge_answer",
            "expected_match_status": "KNOWN",
            "must_not_include": "",
        }
        records.append(
            {
                **base,
                "question": f"{term}이 뭐야",
                "answer": answer,
                "must_include": "일반 설명",
            }
        )
        records.append(
            {
                **base,
                "question": f"{term} 설명해줘",
                "answer": answer,
                "must_include": term,
            }
        )
    return records


def make_meta_records(data_cutoff: str) -> list[dict[str, str]]:
    answer = (
        f"데이터 기준일: {data_cutoff}\n"
        "답변 가능 범위: 아파트 기본정보, 가격, 교통, 정책, 공원/병원 기반 비교\n"
        "예시 질문: 송파구에서 10억 이하 아파트 추천해줘 / 지하철 접근성 좋은 아파트 3개 비교해줘 / 헬리오시티 세대수 알려줘"
    )
    return [
        {
            "question": "이 데이터 언제 기준이야",
            "answer": answer,
            "아파트명": "DATA_SCOPE",
            "문서ID": "META_001",
            "category": "scope_meta",
            "answer_type": "meta_answer",
            "expected_answer_type": "meta_answer",
            "expected_match_status": "KNOWN",
            "must_include": data_cutoff,
            "must_not_include": "",
        },
        {
            "question": "어떤 질문을 하면 잘 답해줘",
            "answer": answer,
            "아파트명": "DATA_SCOPE",
            "문서ID": "META_002",
            "category": "scope_meta",
            "answer_type": "meta_answer",
            "expected_answer_type": "meta_answer",
            "expected_match_status": "KNOWN",
            "must_include": "예시 질문",
            "must_not_include": "",
        },
    ]


def make_no_match_records(data_cutoff: str) -> list[dict[str, str]]:
    answer = (
        f"현재 데이터 기준({data_cutoff})으로 해당 조건에 맞는 아파트를 찾지 못했습니다.\n"
        "조건을 완화하거나 지역/예산 범위를 다시 지정해 주세요."
    )
    return [
        {
            "question": "서울에서 1억 이하이면서 지하철 100m 이내 아파트 추천해줘",
            "answer": answer,
            "아파트명": "NO_MATCH",
            "문서ID": "NO_MATCH_001",
            "category": "recommend_no_match",
            "answer_type": "no_match_response",
            "expected_answer_type": "no_match_response",
            "expected_match_status": "NO_MATCH",
            "must_include": "찾지 못했습니다",
            "must_not_include": "APT_",
        },
        {
            "question": "송파구에서 2억 이하 대단지 신축 아파트 추천해줘",
            "answer": answer,
            "아파트명": "NO_MATCH",
            "문서ID": "NO_MATCH_002",
            "category": "recommend_no_match",
            "answer_type": "no_match_response",
            "expected_answer_type": "no_match_response",
            "expected_match_status": "NO_MATCH",
            "must_include": "조건을 완화",
            "must_not_include": "APT_",
        },
    ]


def make_comparative_records(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty:
        return []
    top_region = safe_text(df["시군구"].dropna().astype(str).value_counts().index[0]) or "서울"
    return [
        {
            "question": f"{top_region}에서 공원 가까운 아파트 비교해줘",
            "answer": "비교 기준: 공원_비교요약\n현재 데이터 기준으로 공원 관련 지표가 있는 후보들을 비교할 수 있습니다.",
            "아파트명": "COMPARATIVE",
            "문서ID": "COMP_001",
            "category": "comparative_supported",
            "answer_type": "comparison_recommendation",
            "expected_answer_type": "comparison_recommendation",
            "expected_match_status": "EXACT_MATCH",
            "must_include": "비교 기준",
            "must_not_include": "",
        },
        {
            "question": f"{top_region}에서 병원 접근성 좋은 아파트 비교해줘",
            "answer": "비교 기준: 병원_비교요약\n현재 데이터 기준으로 병원 접근 지표가 있는 후보들을 비교할 수 있습니다.",
            "아파트명": "COMPARATIVE",
            "문서ID": "COMP_002",
            "category": "comparative_supported",
            "answer_type": "comparison_recommendation",
            "expected_answer_type": "comparison_recommendation",
            "expected_match_status": "EXACT_MATCH",
            "must_include": "비교 기준",
            "must_not_include": "",
        },
        {
            "question": f"{top_region}에서 아이 키우기 좋은 아파트 추천해줘",
            "answer": (
                "현재 MVP에서는 공원, 병원, 지하철처럼 데이터로 직접 판정 가능한 조건만 비교 추천할 수 있습니다.\n"
                "예시 질문: 공원 가까운 아파트 비교해줘 / 병원 접근성 좋은 곳 비교해줘"
            ),
            "아파트명": "COMPARATIVE",
            "문서ID": "COMP_003",
            "category": "comparative_unsupported",
            "answer_type": "unsupported_comparative_response",
            "expected_answer_type": "unsupported_comparative_response",
            "expected_match_status": "UNKNOWN",
            "must_include": "데이터로 직접 판정 가능한 조건",
            "must_not_include": "",
        },
    ]


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


def main() -> None:
    random.seed(RANDOM_SEED)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")
    if not INPUT_KNOWLEDGE.exists():
        raise FileNotFoundError(f"지식 사전 파일이 없습니다: {INPUT_KNOWLEDGE}")

    df, encoding = load_csv(INPUT_FILE)
    knowledge_df, _ = load_csv(INPUT_KNOWLEDGE)
    original_rows = len(df)
    data_cutoff = (
        safe_text(df["데이터기준일"].dropna().iloc[0])
        if "데이터기준일" in df.columns and df["데이터기준일"].notna().any()
        else "기준일 미상"
    )

    duplicated_names = set(
        df.loc[df["아파트명"].duplicated(keep=False) & df["아파트명"].notna(), "아파트명"].astype(str)
    )

    qa_records: list[dict[str, str]] = []
    skipped_null_answers = 0

    for _, row in df.iterrows():
        apartment_name = safe_text(row.get("아파트명"))
        document_id = safe_text(row.get("문서ID"))
        if not apartment_name or not document_id:
            continue

        subject = build_subject(row, duplicated_names)
        for template in build_templates(subject):
            answer_builder = ANSWER_BUILDERS[template["answer_key"]]
            answer = answer_builder(row)
            question = template["question"].strip()
            if not question:
                continue
            if answer is None or is_missing(answer):
                skipped_null_answers += 1
                continue
            qa_records.append(
                {
                    "question": question,
                    "answer": str(answer).strip(),
                    "아파트명": apartment_name,
                    "문서ID": document_id,
                    "category": template["category"],
                    "answer_type": "apartment_fact_lookup",
                    "expected_answer_type": "apartment_fact_lookup",
                    "expected_match_status": "EXACT_MATCH",
                    "must_include": apartment_name,
                    "must_not_include": "",
                }
            )

    qa_records.extend(make_knowledge_records(knowledge_df))
    qa_records.extend(make_meta_records(data_cutoff))
    qa_records.extend(make_no_match_records(data_cutoff))
    qa_records.extend(make_comparative_records(df))

    qa_df = pd.DataFrame(qa_records)
    before_filter_count = len(qa_df)

    qa_df = qa_df.dropna(subset=["question", "answer", "아파트명", "문서ID"])
    qa_df = qa_df[qa_df["question"].astype(str).str.strip() != ""]
    qa_df = qa_df[qa_df["answer"].astype(str).str.strip() != ""]
    qa_df = qa_df[qa_df["아파트명"].astype(str).str.strip() != ""]
    duplicate_question_count = int(qa_df["question"].duplicated().sum())
    qa_df = qa_df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)

    if len(qa_df) < MIN_QA_TARGET:
        raise RuntimeError(f"생성된 QA 수가 목표보다 적습니다. 생성 수={len(qa_df)}, 목표={MIN_QA_TARGET}")

    qa_df.to_csv(OUTPUT_QA, index=False, encoding="utf-8-sig")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fh:
        for record in qa_df.to_dict(orient="records"):
            payload = {
                "instruction": record["question"],
                "input": "",
                "output": record["answer"],
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    eval_size = min(1000, len(qa_df))
    eval_columns = [
        "question",
        "answer",
        "문서ID",
        "expected_answer_type",
        "expected_match_status",
        "must_include",
        "must_not_include",
    ]
    eval_df = qa_df.sample(n=eval_size, random_state=RANDOM_SEED)[eval_columns].rename(
        columns={
            "answer": "expected_answer",
            "문서ID": "expected_doc_id",
        }
    )
    eval_df.to_csv(OUTPUT_EVAL, index=False, encoding="utf-8-sig")

    per_category = qa_df["category"].value_counts().rename_axis("category").reset_index(name="count")
    sample_df = qa_df.head(12)

    report_lines = [
        "# QA Generation Report",
        "",
        "## 개요",
        f"- 입력 파일: `{INPUT_FILE.name}`",
        f"- 입력 인코딩: `{encoding}`",
        f"- 입력 아파트 행 수: {original_rows}",
        f"- 필터 전 QA 수: {before_filter_count}",
        f"- 필터 후 QA 수: {len(qa_df)}",
        f"- null 응답으로 제외된 수: {skipped_null_answers}",
        f"- 질문 중복 제거 수: {duplicate_question_count}",
        f"- 평가셋 크기: {len(eval_df)}",
        "",
        "## 생성 규칙 요약",
        "- 아파트별 사실 QA를 기본으로 생성했습니다.",
        "- 일반 부동산 지식 QA와 데이터 범위 안내 QA를 추가했습니다.",
        "- no-match 응답과 지원/미지원 비교 질문을 별도 카테고리로 추가했습니다.",
        "",
        "## 카테고리별 QA 수",
        markdown_table(per_category),
        "",
        "## 샘플 QA 12개",
        markdown_table(sample_df),
        "",
        "## 생성 파일",
        f"- `{OUTPUT_QA.name}`",
        f"- `{OUTPUT_JSONL.name}`",
        f"- `{OUTPUT_EVAL.name}`",
        f"- `{OUTPUT_REPORT.name}`",
    ]
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
