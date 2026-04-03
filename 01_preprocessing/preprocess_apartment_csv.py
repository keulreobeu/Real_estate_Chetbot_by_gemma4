from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "original"
QA_DIR = DATA_DIR / "qa"
REPORT_DIR = PROJECT_ROOT / "00_Report"
INPUT_FILE = RAW_DIR / "apartment_20230905.csv"
OUTPUT_CLEANED = DATA_DIR / "apartment_chatbot_cleaned.csv"
OUTPUT_QA = QA_DIR / "apartment_chatbot_qa_base.csv"
OUTPUT_REPORT = REPORT_DIR / "01_preprocessing_report.md"
OUTPUT_MAPPING = DATA_DIR / "apartment_column_mapping.csv"
LOG_FILE = REPORT_DIR / "05_preprocessing_log.txt"

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def log(message: str) -> None:
    print(message)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def detect_and_load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error = None
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding)
            return df, encoding
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            log(f"[encoding] failed with {encoding}: {exc}")
    raise RuntimeError(f"CSV 파일을 읽지 못했습니다: {last_error}")


def normalize_name(name: str) -> str:
    normalized = re.sub(r"\s+", "_", str(name).strip())
    normalized = normalized.replace("/", "_")
    normalized = re.sub(r"[()]+", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def make_unique_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique_columns: list[str] = []
    for column in columns:
        base = column
        count = seen.get(base, 0)
        if count:
            unique_columns.append(f"{base}_{count}")
        else:
            unique_columns.append(base)
        seen[base] = count + 1
    return unique_columns


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "None", "NULL", "null"}:
        return True
    return False


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"(만원|원|㎡|m2|m²|평|세대|층|대)", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def parse_address(address: Any) -> dict[str, Any]:
    result = {"시도": pd.NA, "시군구": pd.NA, "동": pd.NA, "상세주소": pd.NA, "주소분리_성공": False}
    if is_missing(address):
        return result

    text = re.sub(r"\s+", " ", str(address).strip())
    tokens = text.split(" ")
    if not tokens:
        return result

    result["시도"] = tokens[0]

    if len(tokens) >= 4 and tokens[1].endswith("시") and tokens[2].endswith(("구", "군")):
        result["시군구"] = f"{tokens[1]} {tokens[2]}"
        result["동"] = tokens[3]
        detail_start = 4
    elif len(tokens) >= 3:
        result["시군구"] = tokens[1]
        result["동"] = tokens[2]
        detail_start = 3
    elif len(tokens) >= 2:
        result["시군구"] = tokens[1]
        detail_start = 2
    else:
        detail_start = 1

    if len(tokens) > detail_start:
        result["상세주소"] = " ".join(tokens[detail_start:])

    has_core = all(not is_missing(result[key]) for key in ["시도", "시군구", "동"])
    result["주소분리_성공"] = bool(has_core)
    return result


def boolish_to_label(value: Any, true_label: str, false_label: str | None = None) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "y", "yes", "적용", "해당"}:
            return true_label
        if lowered in {"false", "0", "n", "no", "미적용", "비해당"}:
            return false_label
    try:
        numeric = float(value)
        if numeric == 1:
            return true_label
        if numeric == 0:
            return false_label
    except Exception:
        pass
    if value is True:
        return true_label
    if value is False:
        return false_label
    return None


def fmt_number(value: Any, digits: int = 0) -> str | None:
    if is_missing(value):
        return None
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if pd.isna(numeric):
        return None
    if digits == 0:
        if numeric.is_integer():
            return f"{int(numeric):,}"
        return f"{numeric:,.0f}"
    return f"{numeric:,.{digits}f}"


def fmt_metric(value: Any, unit: str = "", digits: int = 2) -> str | None:
    number = fmt_number(value, digits=digits)
    if number is None:
        return None
    return f"{number}{unit}"


def build_special_notes(row: pd.Series) -> str:
    items: list[str] = []
    mapping = {
        "분양당시_투기과열지구": "분양 당시 투기과열지구",
        "현재_투기과열지구": "현재 투기과열지구",
        "분양당시_분양가상한제": "분양 당시 분양가상한제 적용",
        "현재_분양가상한제": "현재 분양가상한제 적용",
    }
    for column, label in mapping.items():
        if column in row.index:
            note = boolish_to_label(row[column], label)
            if note:
                items.append(note)
    return "; ".join(items) if items else "규제 특이사항 없음"


def build_location_text(row: pd.Series) -> str | None:
    parts = []
    for column in ["시도", "시군구", "동"]:
        value = row.get(column)
        if not is_missing(value):
            parts.append(str(value))
    if not is_missing(row.get("상세주소")):
        parts.append(str(row["상세주소"]))
    if parts:
        return " ".join(parts)
    if not is_missing(row.get("법정동주소")):
        return str(row["법정동주소"])
    return None


def build_description(row: pd.Series) -> str:
    apartment_name = row.get("아파트명", "이 아파트")
    location = build_location_text(row)
    households = fmt_number(row.get("세대수"))
    exclusive_area = fmt_metric(row.get("전용면적"), "㎡", 2)
    supply_area = fmt_metric(row.get("공급면적"), "㎡", 2)
    price = fmt_number(row.get("공급액(만원)"))
    price_per_pyeong = fmt_number(row.get("평당_공급액"), 2)
    move_in_year = fmt_number(row.get("입주예정연도"))
    builder = "대형 건설사 브랜드 단지" if row.get("대형건설사") is True else None
    subway_distance = fmt_metric(row.get("지하철역_거리"), "km", 2)
    subway_station = row.get("지하철역") if not is_missing(row.get("지하철역")) else row.get("역사명")
    parking = fmt_number(row.get("세대평균_주차대수"), 2)
    notes = row.get("특이사항", "규제 특이사항 없음")

    sentences: list[str] = []
    if location:
        sentences.append(f"{apartment_name}은 {location}에 위치한 아파트 단지입니다.")
    else:
        sentences.append(f"{apartment_name}은 청약 및 분양 정보를 확인할 수 있는 아파트 단지입니다.")

    scale_bits: list[str] = []
    if households:
        scale_bits.append(f"총 {households}세대 규모")
    if exclusive_area:
        scale_bits.append(f"전용면적은 {exclusive_area}")
    if supply_area:
        scale_bits.append(f"공급면적은 {supply_area}")
    if scale_bits:
        sentences.append(", ".join(scale_bits) + "입니다.")

    price_bits: list[str] = []
    if price:
        price_bits.append(f"공급가는 {price}만원 수준")
    if price_per_pyeong:
        price_bits.append(f"평당 공급가는 {price_per_pyeong}만원")
    if move_in_year:
        price_bits.append(f"입주는 {move_in_year}년 예정")
    if price_bits:
        sentences.append(", ".join(price_bits) + "입니다.")

    infra_bits: list[str] = []
    if builder:
        infra_bits.append(builder)
    if subway_station and subway_distance:
        infra_bits.append(f"가장 가까운 지하철역은 {subway_station}이며 직선거리는 약 {subway_distance}")
    if parking:
        infra_bits.append(f"세대평균 주차대수는 {parking}대")
    if infra_bits:
        sentences.append(", ".join(infra_bits) + "입니다.")

    if notes == "규제 특이사항 없음":
        sentences.append("규제 측면에서 별도로 확인된 특이사항은 없습니다.")
    else:
        sentences.append(f"정책 특이사항으로는 {notes}이 있습니다.")

    return " ".join(sentences[:5])


def build_search_keywords(row: pd.Series) -> str:
    keywords: list[str] = []
    for column in [
        "아파트명",
        "시도",
        "시군구",
        "동",
        "타입",
        "전용면적",
        "입주예정연도",
        "지하철역",
        "특이사항",
    ]:
        value = row.get(column)
        if is_missing(value):
            continue
        if isinstance(value, float):
            if pd.isna(value):
                continue
            keywords.append(str(round(value, 2)))
        else:
            keywords.append(str(value))
    deduped: list[str] = []
    for keyword in keywords:
        if keyword not in deduped:
            deduped.append(keyword)
    return ", ".join(deduped)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| 데이터 없음 |\n|---|"
    safe_df = df.copy().fillna("")
    headers = [str(col) for col in safe_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in safe_df.iterrows():
        values = [str(value).replace("\n", " ").replace("|", "/") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_FILE}")

    df_raw, encoding = detect_and_load_csv(INPUT_FILE)
    original_shape = df_raw.shape

    original_columns = [str(col) for col in df_raw.columns]
    cleaned_column_names = make_unique_names([normalize_name(col) for col in original_columns])
    column_mapping = pd.DataFrame(
        {"원본_컬럼명": original_columns, "정리된_컬럼명": cleaned_column_names}
    )

    df = df_raw.copy()
    df.columns = cleaned_column_names

    rename_map = {
        "공급액만원": "공급액(만원)",
        "투기과열지구_before": "분양당시_투기과열지구",
        "투기과열지구_after": "현재_투기과열지구",
        "분양가상한제_before": "분양당시_분양가상한제",
        "분양가상한제_after": "현재_분양가상한제",
    }
    existing_rename_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=existing_rename_map)

    mapping_extra = pd.DataFrame(
        {"원본_컬럼명": list(existing_rename_map.keys()), "정리된_컬럼명": list(existing_rename_map.values())}
    )
    column_mapping = (
        pd.concat([column_mapping, mapping_extra], ignore_index=True)
        .drop_duplicates(subset=["원본_컬럼명", "정리된_컬럼명"])
    )
    column_mapping.to_csv(OUTPUT_MAPPING, index=False, encoding="utf-8-sig")

    numeric_candidates = [
        "위도",
        "경도",
        "세대수",
        "임대세대수",
        "최고층",
        "최저층",
        "최대공급면적",
        "최소공급면적",
        "총아파트동수",
        "용적률",
        "건폐율",
        "세대평균_주차대수",
        "공급면적",
        "전용면적",
        "전용율",
        "방수",
        "욕실수",
        "입주예정연도",
        "공급액만원",
        "공급액(만원)",
        "지하철역_거리",
        "1차병원",
        "2차병원",
        "3차병원",
        "공원",
        "대학",
        "소매",
        "음식",
        "교육",
        "장례식장",
        "보건의료",
        "유원지오락",
        "총인구수",
        "분양당시_투기과열지구",
        "현재_투기과열지구",
        "분양당시_분양가상한제",
        "현재_분양가상한제",
        "시군구내_통근통학",
        "타시군구_통근통학",
        "타시군구_시군구내",
        "평당_공급액",
        "타입",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = clean_numeric_series(df[column])

    if "대형건설사" in df.columns:
        df["대형건설사"] = df["대형건설사"].astype("boolean")

    address_column = first_existing(df, ["법정동주소", "주소", "도로명주소", "지번주소"])
    parsed_records: list[dict[str, Any]] = []
    if address_column:
        parsed_records = [parse_address(value) for value in df[address_column]]
        parsed_df = pd.DataFrame(parsed_records)
    else:
        parsed_df = pd.DataFrame([parse_address(pd.NA) for _ in range(len(df))])

    for column in ["시도", "시군구", "동", "상세주소"]:
        df[column] = parsed_df[column]

    if "광역" in df.columns:
        df["시도"] = df["시도"].where(df["시도"].notna(), df["광역"])
    if "기초" in df.columns:
        df["시군구"] = df["시군구"].where(df["시군구"].notna(), df["기초"])

    df["주소분리_성공"] = parsed_df["주소분리_성공"]
    address_success_count = int(df["주소분리_성공"].fillna(False).sum())
    address_fail_count = int(len(df) - address_success_count)

    if all(col in df.columns for col in ["1차병원", "2차병원", "3차병원"]):
        df["병원"] = df[["1차병원", "2차병원", "3차병원"]].fillna(0).sum(axis=1)
    else:
        df["병원"] = pd.NA

    df["특이사항"] = df.apply(build_special_notes, axis=1)

    key_for_doc = []
    for column in ["아파트명", "시도", "시군구", "동", "타입", "전용면적"]:
        if column in df.columns:
            key_for_doc.append(df[column].astype(str))
    if key_for_doc:
        doc_key = pd.concat(key_for_doc, axis=1).agg("|".join, axis=1)
        df["문서ID"] = doc_key.apply(
            lambda value: "APT-" + hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:24]
        )
    else:
        df["문서ID"] = [f"APT-{idx:06d}" for idx in range(1, len(df) + 1)]

    df["검색키워드"] = df.apply(build_search_keywords, axis=1)
    df["description"] = df.apply(build_description, axis=1)

    duplicate_count = int(df.duplicated().sum())
    df_cleaned = df.drop_duplicates().copy()

    preferred_order = [
        "문서ID",
        "아파트명",
        "시도",
        "시군구",
        "동",
        "상세주소",
        "법정동주소",
        "위도",
        "경도",
        "세대수",
        "총아파트동수",
        "최고층",
        "최저층",
        "공급면적",
        "전용면적",
        "방수",
        "욕실수",
        "세대평균_주차대수",
        "공급액(만원)",
        "평당_공급액",
        "입주예정연도",
        "대형건설사",
        "지하철역",
        "지하철역_거리",
        "공원",
        "대학",
        "병원",
        "분양당시_투기과열지구",
        "현재_투기과열지구",
        "분양당시_분양가상한제",
        "현재_분양가상한제",
        "특이사항",
        "검색키워드",
        "description",
    ]
    ordered_columns = [column for column in preferred_order if column in df_cleaned.columns]
    ordered_columns.extend([column for column in df_cleaned.columns if column not in ordered_columns])
    df_cleaned = df_cleaned[ordered_columns]

    qa_columns = [
        "문서ID",
        "아파트명",
        "시도",
        "시군구",
        "동",
        "전용면적",
        "공급액(만원)",
        "평당_공급액",
        "입주예정연도",
        "특이사항",
        "description",
    ]
    qa_columns = [column for column in qa_columns if column in df_cleaned.columns]
    df_qa = df_cleaned[qa_columns].copy()

    df_cleaned.to_csv(OUTPUT_CLEANED, index=False, encoding="utf-8-sig")
    df_qa.to_csv(OUTPUT_QA, index=False, encoding="utf-8-sig")

    null_summary = (
        df_cleaned.isna().sum().sort_values(ascending=False).reset_index().rename(columns={"index": "컬럼", 0: "결측치수"})
    )
    major_nulls = null_summary[null_summary["결측치수"] > 0].head(20)

    sample_preview = dataframe_to_markdown(df_cleaned.head(5))
    dtype_summary = pd.DataFrame({"컬럼": df_raw.columns.astype(str), "dtype": df_raw.dtypes.astype(str)})
    original_nulls = df_raw.isna().sum().sort_values(ascending=False).reset_index()
    original_nulls.columns = ["컬럼", "결측치수"]

    report_lines = [
        "# Apartment CSV Preprocessing Report",
        "",
        "## 실행 개요",
        f"- 입력 파일: `{INPUT_FILE.name}`",
        f"- 감지 인코딩: `{encoding}`",
        f"- 원본 행/열 수: {original_shape[0]}행 / {original_shape[1]}열",
        f"- 정리 후 행/열 수: {df_cleaned.shape[0]}행 / {df_cleaned.shape[1]}열",
        f"- 제거된 완전 중복 행 수: {duplicate_count}",
        f"- 주소 분리 성공/실패: {address_success_count} / {address_fail_count}",
        "",
        "## 원본 컬럼 목록",
        ", ".join(original_columns),
        "",
        "## 원본 dtype 요약",
        dataframe_to_markdown(dtype_summary),
        "",
        "## 원본 결측치 상위 20개",
        dataframe_to_markdown(original_nulls.head(20)),
        "",
        "## 중복 처리 원칙",
        "- 동일 아파트명이라도 면적 타입이 다르면 유지했습니다.",
        "- `DataFrame.duplicated()` 기준의 완전 중복 행만 제거했습니다.",
        "",
        "## 주소 분리 처리",
        f"- 사용한 주소 컬럼: `{address_column}`" if address_column else "- 사용 가능한 주소 컬럼이 없어 분리를 건너뛰었습니다.",
        "- `시도`, `시군구`, `동`, `상세주소`를 가능한 범위에서 분리했습니다.",
        "- `광역`, `기초` 컬럼이 있을 경우 각각 `시도`, `시군구`의 보조 소스로 활용했습니다.",
        "",
        "## 생성된 새 컬럼 목록",
        ", ".join(
            [
                column
                for column in ["시도", "시군구", "동", "상세주소", "병원", "특이사항", "문서ID", "검색키워드", "description", "주소분리_성공"]
                if column in df_cleaned.columns
            ]
        ),
        "",
        "## 주요 결측치 현황",
        dataframe_to_markdown(major_nulls) if not major_nulls.empty else "결측치가 없습니다.",
        "",
        "## 샘플 5행 미리보기",
        sample_preview,
        "",
        "## 산출물",
        f"- `{OUTPUT_CLEANED.name}`",
        f"- `{OUTPUT_QA.name}`",
        f"- `{OUTPUT_REPORT.name}`",
        f"- `{OUTPUT_MAPPING.name}`",
        f"- `{LOG_FILE.name}`",
    ]
    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    log("[done] preprocessing complete")
    log(f"[done] cleaned file: {OUTPUT_CLEANED}")
    log(f"[done] qa base file: {OUTPUT_QA}")
    log(f"[done] report file: {OUTPUT_REPORT}")
    log(f"[done] mapping file: {OUTPUT_MAPPING}")
    log(f"[summary] original shape={original_shape}, cleaned shape={df_cleaned.shape}, duplicates_removed={duplicate_count}")
    log(f"[summary] address split success={address_success_count}, fail={address_fail_count}")


if __name__ == "__main__":
    main()
