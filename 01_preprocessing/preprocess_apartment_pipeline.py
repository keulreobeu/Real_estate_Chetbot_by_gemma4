from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "original"
QA_DIR = DATA_DIR / "qa"
REPORT_DIR = PROJECT_ROOT / "00_Report"
INPUT_CANDIDATES = [
    RAW_DIR / "apartment_20230905.csv",
]

OUTPUT_MAIN = DATA_DIR / "apartment_chatbot_v3.csv"
OUTPUT_QA = QA_DIR / "apartment_chatbot_qa_base_v3.csv"
OUTPUT_STATION_LINE = DATA_DIR / "station_line_map.csv"
OUTPUT_APT_STATION = DATA_DIR / "apartment_station_map.csv"
OUTPUT_MAPPING = DATA_DIR / "apartment_column_mapping.csv"
OUTPUT_REPORT = REPORT_DIR / "02_preprocessing_full_report.md"

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "입력 파일을 찾지 못했습니다. apartment_20230905.csv 또는 apartment_20230905 - apartment_20230905.csv.csv 파일이 필요합니다."
    )


def detect_and_load_csv(path: Path) -> tuple[pd.DataFrame, str]:
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


def parse_address(address: Any) -> dict[str, Any]:
    result = {"시도": pd.NA, "시군구": pd.NA, "동": pd.NA, "상세주소": pd.NA, "주소분리_성공": False}
    if is_missing(address):
        return result

    text = " ".join(str(address).strip().split())
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

    result["주소분리_성공"] = all(not is_missing(result[key]) for key in ["시도", "시군구", "동"])
    return result


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"(만원|원|㎡|m2|m²|평|세대|층|km|m)", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def to_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "y", "yes", "적용", "해당"}:
            return True
        if lowered in {"false", "0", "n", "no", "미적용", "비해당"}:
            return False
    try:
        numeric = float(value)
        if numeric == 1:
            return True
        if numeric == 0:
            return False
    except Exception:
        return None
    return None


def format_number(value: Any, digits: int = 0) -> str | None:
    if is_missing(value):
        return None
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if pd.isna(numeric):
        return None
    if digits == 0:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def format_area(value: Any) -> str | None:
    num = format_number(value, 2)
    return f"{num}㎡" if num else None


def parse_line_list(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            raw = parsed if isinstance(parsed, list) else [text]
        except Exception:
            raw = [piece.strip() for piece in text.split(",")]
    lines = []
    for item in raw:
        line = str(item).strip().strip("'").strip('"')
        if line and line.lower() != "nan" and line not in lines:
            lines.append(line)
    return lines


def summarize_medical(row: pd.Series) -> str:
    primary = row.get("1차병원")
    secondary = row.get("2차병원")
    tertiary = row.get("3차병원")
    primary_n = 0 if pd.isna(primary) else int(primary)
    secondary_n = 0 if pd.isna(secondary) else int(secondary)
    tertiary_n = 0 if pd.isna(tertiary) else int(tertiary)
    total = primary_n + secondary_n + tertiary_n
    if total == 0:
        return "의료시설 정보 없음"
    if tertiary_n > 0 and secondary_n > 0:
        return "1차·2차 병원이 주변에 분포해 있으며 상급 의료기관 접근도 가능한 지역입니다."
    if tertiary_n > 0:
        return "상급 의료기관 접근이 가능한 지역입니다."
    if secondary_n > 0:
        return "중형급 이상 의료시설 접근이 가능한 생활권입니다."
    return "기초 의료시설을 중심으로 이용 가능한 지역입니다."


def summarize_living_infra(total: float, q1: float, q3: float) -> str:
    if pd.isna(total):
        return "생활 인프라 정보 없음"
    if total >= q3:
        return "주변 상권과 생활 편의시설이 비교적 풍부한 지역입니다."
    if total <= q1:
        return "생활 인프라는 제한적인 편입니다."
    return "생활 편의시설이 무난하게 형성된 지역입니다."


def summarize_commute(row: pd.Series) -> str:
    inside = row.get("시군구내_통근통학")
    outside = row.get("타시군구_통근통학")
    if pd.isna(inside) and pd.isna(outside):
        return "통근통학 정보 없음"
    inside = 0 if pd.isna(inside) else float(inside)
    outside = 0 if pd.isna(outside) else float(outside)
    if outside > inside:
        return "외부 지역으로 출퇴근하는 생활권 특성이 있습니다."
    if inside > outside:
        return "지역 내 생활권 중심 이동이 많은 지역입니다."
    return "지역 내외 이동 수요가 비교적 균형적인 지역입니다."


def classify_area(exclusive_area: Any) -> str:
    if is_missing(exclusive_area):
        return "면적 정보 없음"
    area = float(exclusive_area)
    if area < 40:
        return "소형"
    if area < 60:
        return "중소형"
    if area < 85:
        return "중형"
    if area < 120:
        return "중대형"
    return "대형"


def summarize_structure(row: pd.Series) -> str:
    area = format_number(row.get("전용면적"), 0)
    rooms = format_number(row.get("방수"), 0)
    baths = format_number(row.get("욕실수"), 0)
    hallway = row.get("현관구조")

    bits = []
    if area:
        bits.append(f"전용면적 {area}㎡ 기준")
    if rooms:
        bits.append(f"방 {rooms}개")
    if baths:
        bits.append(f"욕실 {baths}개")
    if not is_missing(hallway):
        bits.append(f"{hallway} 현관구조")
    if not bits:
        return "구조 정보 없음"
    return " ".join(bits) + " 구조입니다."


def summarize_price(price_per_pyeong: Any, low: float, high: float) -> str:
    if is_missing(price_per_pyeong):
        return "가격 정보 없음"
    value = float(price_per_pyeong)
    if value >= high:
        return "평당 공급가가 비교적 높은 편입니다."
    if value <= low:
        return "평당 공급가가 비교적 합리적인 편입니다."
    return "평당 공급가가 평균 수준입니다."


def summarize_builder(value: Any) -> str:
    is_major = to_bool(value)
    if is_major is True:
        return "대형 건설사가 시공한 단지입니다."
    if is_major is False:
        return "중견 건설사 단지입니다."
    return "건설사 정보 확인이 추가로 필요한 단지입니다."


def summarize_policy(row: pd.Series) -> str:
    labels = []
    mapping = {
        "분양당시_투기과열지구": "분양 당시 투기과열지구",
        "현재_투기과열지구": "현재 투기과열지구",
        "분양당시_분양가상한제": "분양 당시 분양가상한제 적용",
        "현재_분양가상한제": "현재 분양가상한제 적용",
    }
    for column, label in mapping.items():
        if to_bool(row.get(column)) is True:
            labels.append(label)
    if not labels:
        return "정책상 특이사항은 확인되지 않았습니다."
    if len(labels) == 1:
        return f"{labels[0]} 단지입니다."
    if len(labels) == 2:
        return f"{labels[0]} 및 {labels[1]} 단지입니다."
    return f"{', '.join(labels[:-1])} 및 {labels[-1]} 단지입니다."


def build_description(row: pd.Series) -> str:
    name = row.get("아파트명", "이 단지")
    location_parts = [row.get("시도"), row.get("시군구"), row.get("동")]
    location = " ".join(str(v) for v in location_parts if not is_missing(v))
    households = format_number(row.get("세대수"), 0)
    area = format_area(row.get("전용면적"))
    station = row.get("가장가까운역")
    distance_m = row.get("거리_m")
    price = format_number(row.get("공급액(만원)"), 0)

    sentences = []
    if location:
        sentences.append(f"{name}은 {location}에 위치한 아파트 단지입니다.")
    else:
        sentences.append(f"{name}은 청약 아파트 정보가 정리된 단지입니다.")

    size_bits = []
    if households:
        size_bits.append(f"총 {households}세대 규모")
    if area:
        size_bits.append(f"전용면적은 {area}")
    if size_bits:
        sentences.append(", ".join(size_bits) + "입니다.")

    traffic_bits = []
    if not is_missing(station):
        traffic_bits.append(f"가장 가까운 지하철역은 {station}")
    if not pd.isna(distance_m):
        traffic_bits.append(f"약 {int(round(float(distance_m))):,}m 거리")
    if traffic_bits:
        sentences.append(", ".join(traffic_bits) + "입니다.")

    price_bits = []
    if price:
        price_bits.append(f"공급가는 {price}만원 수준")
    if not is_missing(row.get("가격요약")):
        price_bits.append(str(row.get("가격요약")))
    if price_bits:
        sentences.append(", ".join(price_bits))

    for extra in [
        row.get("의료시설_요약"),
        row.get("생활인프라_요약"),
        row.get("정책특이사항_설명"),
    ]:
        if not is_missing(extra):
            sentences.append(str(extra))

    return " ".join(sentences)


def build_search_keywords(row: pd.Series) -> str:
    parts: list[str] = []
    for value in [
        row.get("아파트명"),
        row.get("시도"),
        row.get("시군구"),
        row.get("동"),
        row.get("가장가까운역"),
        row.get("가장가까운역_호선요약"),
        row.get("건설사_요약"),
        row.get("면적대"),
        row.get("정책특이사항_설명"),
        row.get("가격요약"),
    ]:
        if is_missing(value):
            continue
        text = str(value).strip()
        if text and text not in parts:
            parts.append(text)
    return ", ".join(parts)


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
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = find_input_file()
    df_raw, encoding = detect_and_load_csv(input_path)
    original_shape = df_raw.shape
    original_columns = list(df_raw.columns)
    original_nulls = df_raw.isna().sum().sort_values(ascending=False)
    duplicate_count = int(df_raw.duplicated().sum())

    rename_map = {
        "투기과열지구_before": "분양당시_투기과열지구",
        "투기과열지구_after": "현재_투기과열지구",
        "분양가상한제_before": "분양당시_분양가상한제",
        "분양가상한제_after": "현재_분양가상한제",
    }

    mapping_records = []
    for col in original_columns:
        mapping_records.append({"원본_컬럼명": col, "변경_컬럼명": rename_map.get(col, col)})
    pd.DataFrame(mapping_records).to_csv(OUTPUT_MAPPING, index=False, encoding="utf-8-sig")

    df = df_raw.drop_duplicates().copy().rename(columns=rename_map)

    numeric_columns = [
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
        "타시군구/시군구내",
        "평당_공급액",
        "타입",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    address_source = "법정동주소" if "법정동주소" in df.columns else None
    parsed = [parse_address(value) for value in (df[address_source] if address_source else [pd.NA] * len(df))]
    parsed_df = pd.DataFrame(parsed)
    for col in ["시도", "시군구", "동", "상세주소", "주소분리_성공"]:
        df[col] = parsed_df[col]

    if "광역" in df.columns:
        df["시도"] = df["시도"].where(df["시도"].notna(), df["광역"])
    if "기초" in df.columns:
        df["시군구"] = df["시군구"].where(df["시군구"].notna(), df["기초"])

    df.insert(0, "문서ID", [f"APT_{idx:06d}" for idx in range(1, len(df) + 1)])

    station_name_source = "역사명" if "역사명" in df.columns else "지하철역"
    df["가장가까운역"] = df[station_name_source] if station_name_source in df.columns else pd.NA

    if "지하철역_거리" in df.columns:
        df["거리_m"] = df["지하철역_거리"].apply(
            lambda x: pd.NA if pd.isna(x) else round(float(x) * 1000, 1)
        )
    else:
        df["거리_m"] = pd.NA

    line_lists = df["노선명_리스트"].apply(parse_line_list) if "노선명_리스트" in df.columns else pd.Series([[]] * len(df))
    df["호선수"] = line_lists.apply(len)
    df["환승역여부"] = df["호선수"].apply(lambda x: "예" if pd.notna(x) and int(x) >= 2 else "아니오")
    df["가장가까운역_호선요약"] = line_lists.apply(lambda items: ", ".join(items) if items else "호선 정보 없음")

    station_line_records = []
    for station, lines in zip(df["가장가까운역"], line_lists):
        if is_missing(station):
            continue
        for line in lines:
            station_line_records.append({"역명": station, "호선": line})
    station_line_df = pd.DataFrame(station_line_records).drop_duplicates()
    station_line_df.to_csv(OUTPUT_STATION_LINE, index=False, encoding="utf-8-sig")

    apt_station_df = pd.DataFrame(
        {
            "문서ID": df["문서ID"],
            "아파트명": df["아파트명"] if "아파트명" in df.columns else pd.NA,
            "역명": df["가장가까운역"],
            "거리_m": df["거리_m"],
            "가장가까운역여부": "Y",
        }
    )
    apt_station_df.to_csv(OUTPUT_APT_STATION, index=False, encoding="utf-8-sig")

    living_cols = [col for col in ["소매", "음식", "교육", "보건의료", "유원지오락"] if col in df.columns]
    living_score = df[living_cols].fillna(0).sum(axis=1) if living_cols else pd.Series([pd.NA] * len(df))
    living_q1 = float(living_score.quantile(0.33)) if living_cols else float("nan")
    living_q3 = float(living_score.quantile(0.67)) if living_cols else float("nan")

    price_low = float(df["평당_공급액"].quantile(0.33)) if "평당_공급액" in df.columns else float("nan")
    price_high = float(df["평당_공급액"].quantile(0.67)) if "평당_공급액" in df.columns else float("nan")

    df["의료시설_요약"] = df.apply(summarize_medical, axis=1)
    df["생활인프라_요약"] = living_score.apply(lambda x: summarize_living_infra(x, living_q1, living_q3))
    df["통근통학_요약"] = df.apply(summarize_commute, axis=1)
    df["구조요약"] = df.apply(summarize_structure, axis=1)
    df["면적대"] = df["전용면적"].apply(classify_area) if "전용면적" in df.columns else "면적 정보 없음"
    df["가격요약"] = (
        df["평당_공급액"].apply(lambda x: summarize_price(x, price_low, price_high))
        if "평당_공급액" in df.columns
        else "가격 정보 없음"
    )
    df["건설사_요약"] = df["대형건설사"].apply(summarize_builder) if "대형건설사" in df.columns else "건설사 정보 없음"
    df["정책특이사항_설명"] = df.apply(summarize_policy, axis=1)
    df["description"] = df.apply(build_description, axis=1)
    df["검색키워드"] = df.apply(build_search_keywords, axis=1)

    rag_columns = [
        "가장가까운역",
        "환승역여부",
        "호선수",
        "가장가까운역_호선요약",
        "거리_m",
        "의료시설_요약",
        "생활인프라_요약",
        "통근통학_요약",
        "구조요약",
        "면적대",
        "가격요약",
        "건설사_요약",
        "정책특이사항_설명",
        "검색키워드",
    ]

    df.to_csv(OUTPUT_MAIN, index=False, encoding="utf-8-sig")

    qa_columns = [
        "아파트명",
        "시도",
        "시군구",
        "동",
        "전용면적",
        "공급액(만원)",
        "평당_공급액",
        "면적대",
        "가격요약",
        "정책특이사항_설명",
        "description",
    ]
    available_qa_columns = [col for col in qa_columns if col in df.columns]
    df[available_qa_columns].to_csv(OUTPUT_QA, index=False, encoding="utf-8-sig")

    sample_cols = [
        "문서ID",
        "아파트명",
        "시도",
        "시군구",
        "동",
        "가장가까운역",
        "가장가까운역_호선요약",
        "면적대",
        "가격요약",
        "정책특이사항_설명",
    ]
    sample_cols = [col for col in sample_cols if col in df.columns]
    sample_df = df[sample_cols].head(10)

    description_example = df[["아파트명", "description"]].head(3)
    address_success = int((df["주소분리_성공"] == True).sum())

    report = [
        "# Apartment Preprocessing Full Report",
        "",
        "## 개요",
        f"- 입력 파일: `{input_path.name}`",
        f"- 감지 인코딩: `{encoding}`",
        f"- 원본 데이터 크기: {original_shape[0]}행 / {original_shape[1]}열",
        f"- 정리 후 데이터 크기: {df.shape[0]}행 / {df.shape[1]}열",
        f"- 중복 제거 수: {duplicate_count}",
        f"- 주소 분리 성공률: {address_success}/{len(df)} ({address_success / len(df) * 100:.2f}%)",
        "",
        "## 원본 컬럼 목록",
        ", ".join(original_columns),
        "",
        "## 원본 결측치 현황 상위 20개",
        markdown_table(
            original_nulls.reset_index().rename(columns={"index": "컬럼", 0: "결측치수"}).head(20)
        ),
        "",
        "## 생성된 컬럼 목록",
        ", ".join(["문서ID", "시도", "시군구", "동", "상세주소", "주소분리_성공"] + rag_columns),
        "",
        "## 샘플 데이터 10행",
        markdown_table(sample_df),
        "",
        "## Description 예시",
        markdown_table(description_example),
        "",
        "## 생성 파일",
        f"- `{OUTPUT_MAIN.name}`",
        f"- `{OUTPUT_QA.name}`",
        f"- `{OUTPUT_STATION_LINE.name}`",
        f"- `{OUTPUT_APT_STATION.name}`",
        f"- `{OUTPUT_MAPPING.name}`",
        f"- `{OUTPUT_REPORT.name}`",
    ]
    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
