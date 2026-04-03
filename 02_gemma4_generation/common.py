from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
QA_DIR = DATA_DIR / "qa"
EVAL_DIR = DATA_DIR / "eval"
REPORT_DIR = PROJECT_ROOT / "00_Report"
PROMPT_FILE = BASE_DIR / "prompts" / "grounded_answer_prompt.txt"
CONFIG_DIR = BASE_DIR / "config"
GENERATION_CONFIG_FILE = CONFIG_DIR / "generation_defaults.json"
LOCAL_MODEL_CONFIG_FILE = CONFIG_DIR / "models.local.json"
EXAMPLE_MODEL_CONFIG_FILE = CONFIG_DIR / "models.local.example.json"

INPUT_MAIN = DATA_DIR / "apartment_chatbot_v3.csv"
INPUT_EVAL = QA_DIR / "evaluation_dataset.csv"
INPUT_EDGE = QA_DIR / "edge_case_eval.csv"

OUTPUT_SOURCE_INDEX = EVAL_DIR / "gemma4_generation_source_index.csv"

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
DEFAULT_MODEL_ID = "gemma4_2b"
DEFAULT_BACKEND = "mock"
DEFAULT_FALLBACK_ANSWER = "데이터에서 확인되지 않습니다."

REQUIRED_MAIN_COLUMNS = [
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
]

REQUIRED_EVAL_COLUMNS = ["question", "expected_answer", "문서ID"]
REQUIRED_EDGE_COLUMNS = ["question", "expected_doc", "expected_field"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV를 읽을 수 없습니다: {path} ({last_error})")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name}에 필요한 컬럼이 없습니다: {missing}")


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "None", "NULL", "null"}:
        return True
    return False


def safe_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return str(value).strip()


def format_number(value: Any, digits: int = 0) -> str:
    if is_missing(value):
        return ""
    try:
        numeric = float(value)
    except Exception:
        return safe_text(value)
    if digits == 0:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^0-9a-z가-힣\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {token for token in normalized.split(" ") if token}


def build_search_text(row: pd.Series) -> str:
    fields = [
        "아파트명",
        "시도",
        "시군구",
        "동",
        "가장가까운역",
        "가장가까운역_호선요약",
        "description",
        "검색키워드",
        "환승역여부",
    ]
    return " ".join(safe_text(row.get(field)) for field in fields if safe_text(row.get(field)))


def infer_used_fields(question: str) -> list[str]:
    normalized = normalize_text(question)
    mappings = [
        (["지하철", "역", "노선", "환승"], ["가장가까운역", "거리_m", "가장가까운역_호선요약", "환승역여부"]),
        (["가격", "공급가", "분양가", "평당"], ["공급액(만원)", "평당_공급액"]),
        (["정책", "분양가상한제", "투기과열지구"], ["정책특이사항_설명"]),
        (["위치", "주소", "어디"], ["시도", "시군구", "동"]),
        (["면적", "전용", "공급"], ["전용면적", "공급면적"]),
    ]
    used_fields: list[str] = []
    for keywords, fields in mappings:
        if any(keyword in normalized for keyword in keywords):
            for field in fields:
                if field not in used_fields:
                    used_fields.append(field)
    if not used_fields:
        used_fields = ["description"]
    return used_fields


def score_row(question: str, row: pd.Series) -> float:
    question_tokens = tokenize(question)
    row_tokens = tokenize(safe_text(row.get("검색텍스트")))
    if not question_tokens or not row_tokens:
        return 0.0

    overlap = question_tokens & row_tokens
    score = float(len(overlap))

    apartment_name = safe_text(row.get("아파트명"))
    if apartment_name and apartment_name in question:
        score += 5.0

    station = safe_text(row.get("가장가까운역"))
    if station and station in question:
        score += 3.0

    district = safe_text(row.get("시군구"))
    if district and district in question:
        score += 2.0

    return score


def retrieve_top_k(question: str, source_df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    scored = source_df.copy()
    scored["retrieval_score"] = scored.apply(lambda row: score_row(question, row), axis=1)
    scored = scored.sort_values(["retrieval_score", "문서ID"], ascending=[False, True])
    return scored.head(top_k).reset_index(drop=True)


def build_context_block(retrieved_df: pd.DataFrame) -> str:
    blocks: list[str] = []
    for _, row in retrieved_df.iterrows():
        block = [
            f"문서ID: {safe_text(row.get('문서ID'))}",
            f"아파트명: {safe_text(row.get('아파트명'))}",
            f"위치: {' '.join(part for part in [safe_text(row.get('시도')), safe_text(row.get('시군구')), safe_text(row.get('동'))] if part)}",
            f"교통: 가장 가까운 역 {safe_text(row.get('가장가까운역'))}, 거리 {format_number(row.get('거리_m'))}m, 노선 {safe_text(row.get('가장가까운역_호선요약'))}, 환승 {safe_text(row.get('환승역여부'))}",
            f"가격: 공급액 {format_number(row.get('공급액(만원)'))}만원, 평당 공급액 {format_number(row.get('평당_공급액'), 2)}만원",
            f"설명: {safe_text(row.get('description'))}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def load_prompt_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"프롬프트 템플릿이 없습니다: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def build_prompt(question: str, retrieved_df: pd.DataFrame) -> str:
    template = load_prompt_template()
    context = build_context_block(retrieved_df)
    return template.format(question=question, context=context)


def append_citation(answer: str, cited_doc_ids: list[str]) -> str:
    if not cited_doc_ids:
        return answer
    citation = f"근거 문서: {', '.join(cited_doc_ids)}"
    if citation in answer:
        return answer
    return f"{answer.rstrip()} {citation}".strip()


def fallback_answer(cited_doc_ids: list[str]) -> str:
    return append_citation(DEFAULT_FALLBACK_ANSWER, cited_doc_ids)


def get_prediction_output_path(mode: str, model_id: str) -> Path:
    return EVAL_DIR / f"gemma4_generation_{mode}_predictions_{model_id}.csv"


def get_metrics_output_path(mode: str, model_id: str) -> Path:
    return EVAL_DIR / f"gemma4_generation_{mode}_metrics_{model_id}.json"


def get_compare_report_path() -> Path:
    return REPORT_DIR / "07_gemma4_generation_model_comparison.md"


def load_generation_config() -> dict[str, Any]:
    if not GENERATION_CONFIG_FILE.exists():
        raise FileNotFoundError(f"생성 설정 파일이 없습니다: {GENERATION_CONFIG_FILE}")
    return load_json(GENERATION_CONFIG_FILE)


def load_model_catalog() -> dict[str, Any]:
    config_path = LOCAL_MODEL_CONFIG_FILE if LOCAL_MODEL_CONFIG_FILE.exists() else EXAMPLE_MODEL_CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f"모델 설정 파일이 없습니다: {LOCAL_MODEL_CONFIG_FILE} 또는 {EXAMPLE_MODEL_CONFIG_FILE}"
        )
    return load_json(config_path)


def expand_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(raw_path)
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def resolve_model_config(model_id: str) -> dict[str, Any]:
    catalog = load_model_catalog()
    models = catalog.get("models", {})
    if model_id not in models:
        raise ValueError(f"지원하지 않는 model_id 입니다: {model_id}. 사용 가능: {sorted(models)}")

    model_config = dict(models[model_id])
    model_config.setdefault("model_id", model_id)
    model_config["model_path"] = str(expand_path(str(model_config["model_path"])))
    return model_config


def pick_default_model_id() -> str:
    catalog = load_model_catalog()
    return str(catalog.get("default_model_id", DEFAULT_MODEL_ID))
