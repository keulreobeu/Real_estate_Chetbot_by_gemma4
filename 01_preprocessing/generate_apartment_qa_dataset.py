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
    raise RuntimeError(f"CSV를 읽지 못했습니다: {last_error}")


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "None", "NULL", "null"}:
        return True
    return False


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


def safe_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value).strip()


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
    return f"{name}는 총 {households}세대 규모의 아파트 단지입니다."


def make_answer_structure(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    structure = safe_text(row.get("구조요약"))
    if not name or not structure:
        return None
    return f"{name}의 구조는 {structure}"


def make_answer_area_band(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    band = safe_text(row.get("면적대"))
    if not name or not band:
        return None
    return f"{name}는 {band} 면적대로 분류됩니다."


def make_answer_location(row: pd.Series) -> str | None:
    name = safe_text(row.get("아파트명"))
    location = build_location(row)
    if not name or not location:
        return None
    return f"{name}는 {location}에 위치한 아파트입니다."


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
        return f"가장 가까운 지하철역은 {station}이며 약 {distance}m 거리에 있습니다."
    return f"가장 가까운 지하철역은 {station}입니다."


def make_answer_transport_summary(row: pd.Series) -> str | None:
    station = safe_text(row.get("가장가까운역"))
    lines = safe_text(row.get("가장가까운역_호선요약"))
    transfer = safe_text(row.get("환승역여부"))
    if not station:
        return None
    transfer_text = "환승역입니다" if transfer == "예" else "환승역은 아닙니다"
    if lines and lines != "호선 정보 없음":
        return f"{station} 이용이 가능하며 연결 노선은 {lines}이고 {transfer_text}."
    return f"{station} 이용이 가능하며 {transfer_text}."


def make_answer_distance(row: pd.Series) -> str | None:
    distance = fmt_int(row.get("거리_m"))
    if not distance:
        return None
    return f"지하철역까지 거리는 약 {distance}m입니다."


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
    return f"평당 공급가는 {value}만원입니다."


def make_answer_price_summary(row: pd.Series) -> str | None:
    summary = safe_text(row.get("가격요약"))
    if not summary:
        return None
    return summary


def make_answer_medical(row: pd.Series) -> str | None:
    summary = safe_text(row.get("의료시설_요약"))
    if not summary:
        return None
    return summary


def make_answer_lifestyle(row: pd.Series) -> str | None:
    summary = safe_text(row.get("생활인프라_요약"))
    if not summary:
        return None
    return summary


def make_answer_commute(row: pd.Series) -> str | None:
    summary = safe_text(row.get("통근통학_요약"))
    if not summary:
        return None
    return summary


def make_answer_builder(row: pd.Series) -> str | None:
    summary = safe_text(row.get("건설사_요약"))
    if not summary:
        return None
    return summary


def make_answer_policy(row: pd.Series) -> str | None:
    summary = safe_text(row.get("정책특이사항_설명"))
    if not summary:
        return None
    return summary


def make_answer_description(row: pd.Series) -> str | None:
    desc = safe_text(row.get("description"))
    if not desc:
        return None
    return desc


def build_templates(subject: str) -> list[dict[str, str]]:
    return [
        {"category": "fact", "question": f"{subject} 전용면적은 얼마야", "answer_key": "fact_area"},
        {"category": "fact", "question": f"{subject} 공급면적은 얼마야", "answer_key": "supply_area"},
        {"category": "fact", "question": f"{subject} 세대수는 몇 세대야", "answer_key": "households"},
        {"category": "fact", "question": f"{subject} 구조는 어떻게 돼", "answer_key": "structure"},
        {"category": "fact", "question": f"{subject} 면적대는 뭐야", "answer_key": "area_band"},
        {"category": "location", "question": f"{subject} 위치가 어디야", "answer_key": "location"},
        {"category": "location", "question": f"{subject} 주소 알려줘", "answer_key": "location_detail"},
        {"category": "location", "question": f"{subject} 어느 지역 아파트야", "answer_key": "location"},
        {"category": "transport", "question": f"{subject} 근처 지하철역은 어디야", "answer_key": "station"},
        {"category": "transport", "question": f"{subject}에서 가장 가까운 역은", "answer_key": "station"},
        {"category": "transport", "question": f"{subject} 지하철 접근성 어때", "answer_key": "transport_summary"},
        {"category": "transport", "question": f"{subject} 가까운 역까지 거리는 얼마야", "answer_key": "distance"},
        {"category": "price", "question": f"{subject} 분양가는 얼마야", "answer_key": "price"},
        {"category": "price", "question": f"{subject} 평당 가격은", "answer_key": "price_per_pyeong"},
        {"category": "price", "question": f"{subject} 가격 수준은 어떤 편이야", "answer_key": "price_summary"},
        {"category": "price", "question": f"{subject} 가격 메리트 있어", "answer_key": "price_summary"},
        {"category": "lifestyle", "question": f"{subject} 주변 상권은 어때", "answer_key": "lifestyle"},
        {"category": "lifestyle", "question": f"{subject} 생활 인프라는 어떤 편이야", "answer_key": "lifestyle"},
        {"category": "lifestyle", "question": f"{subject} 병원 접근성은 어때", "answer_key": "medical"},
        {"category": "lifestyle", "question": f"{subject} 통근 통학 여건은 어때", "answer_key": "commute"},
        {"category": "lifestyle", "question": f"{subject} 건설사 수준은 어때", "answer_key": "builder"},
        {"category": "policy", "question": f"{subject}은 투기과열지구야", "answer_key": "policy"},
        {"category": "policy", "question": f"{subject}은 분양가 상한제 적용됐어", "answer_key": "policy"},
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

    df, encoding = load_csv(INPUT_FILE)
    original_rows = len(df)

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
        templates = build_templates(subject)

        for template in templates:
            answer_builder = ANSWER_BUILDERS[template["answer_key"]]
            answer = answer_builder(row)
            question = template["question"].strip()
            if not question or not apartment_name:
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
                }
            )

    qa_df = pd.DataFrame(qa_records)
    before_filter_count = len(qa_df)

    qa_df = qa_df.dropna(subset=["question", "answer", "아파트명", "문서ID"])
    qa_df = qa_df[qa_df["question"].astype(str).str.strip() != ""]
    qa_df = qa_df[qa_df["answer"].astype(str).str.strip() != ""]
    qa_df = qa_df[qa_df["아파트명"].astype(str).str.strip() != ""]
    duplicate_question_count = int(qa_df["question"].duplicated().sum())
    qa_df = qa_df.drop_duplicates(subset=["question"], keep="first").reset_index(drop=True)

    if len(qa_df) < MIN_QA_TARGET:
        raise RuntimeError(
            f"생성된 QA 수가 목표보다 적습니다. 생성 수={len(qa_df)}, 목표={MIN_QA_TARGET}"
        )

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
    eval_df = qa_df.sample(n=eval_size, random_state=RANDOM_SEED)[["question", "answer", "문서ID"]].rename(
        columns={"answer": "expected_answer"}
    )
    eval_df.to_csv(OUTPUT_EVAL, index=False, encoding="utf-8-sig")

    per_category = qa_df["category"].value_counts().rename_axis("category").reset_index(name="count")
    sample_df = qa_df.head(10)

    report_lines = [
        "# QA Generation Report",
        "",
        "## 개요",
        f"- 입력 파일: `{INPUT_FILE.name}`",
        f"- 감지 인코딩: `{encoding}`",
        f"- 입력 아파트 행 수: {original_rows}",
        f"- 필터 전 QA 수: {before_filter_count}",
        f"- 필터 후 QA 수: {len(qa_df)}",
        f"- null 답변으로 제외된 수: {skipped_null_answers}",
        f"- 질문 중복 제거 수: {duplicate_question_count}",
        f"- 평가셋 크기: {len(eval_df)}",
        "",
        "## 질문 생성 전략",
        "- 각 행마다 25개의 템플릿 질문을 생성했습니다.",
        "- 중복 아파트명은 전용면적/타입 정보를 질문 주어에 포함해 질문 중복을 줄였습니다.",
        "- 답변은 모두 `apartment_chatbot_v3.csv`의 컬럼 값만 사용해 생성했습니다.",
        "",
        "## 카테고리별 QA 수",
        markdown_table(per_category),
        "",
        "## 샘플 QA 10개",
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
