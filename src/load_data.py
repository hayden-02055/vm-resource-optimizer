#!/usr/bin/env python3
"""GWA-T-12 트레이스 로딩 스크립트.

설계안: docs/GWA_T12_데이터_로딩_설계안.md (rev.2)

원칙
  - 원본 11개 컬럼 보존 + vm_id 추가 = 12개 컬럼
  - 값 변환/보간/정렬/이상치 처리를 하지 않는다 (로딩 전용)
  - 파일별 구분자 판별 -> 헤더 검증 -> 청크 단위 읽기 -> 단일 CSV 적재
  - 결측·행수 불일치는 조용히 넘기지 않고 오류로 중단한다
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

TRACE_NAME = "fastStorage"
CHUNKSIZE = 100_000

COLUMNS = [
    "Timestamp [ms]",
    "CPU cores",
    "CPU capacity provisioned [MHZ]",
    "CPU usage [MHZ]",
    "CPU usage [%]",
    "Memory capacity provisioned [KB]",
    "Memory usage [KB]",
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]
INT_COLUMNS = ["Timestamp [ms]", "CPU cores"]

DTYPES = {col: "float64" for col in COLUMNS}
for _col in INT_COLUMNS:
    DTYPES[_col] = "Int64"

# 설계안 §4-1: 원본은 확장자 없는 파일로도 배포된다.
DATA_SUFFIXES = {"", ".csv", ".txt", ".tsv"}
DELIMITER_LABELS = {";": "semicolon+tab", "\t": "tab"}


def sort_key(path: Path):
    """1.csv ... 1250.csv 를 번호순으로, 비숫자 파일명은 뒤에 사전순으로."""
    stem = path.stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def vm_id_of(path: Path):
    stem = path.stem
    return int(stem) if stem.isdigit() else stem


def count_data_rows(path: Path) -> int:
    """pandas와 독립적으로 원본 데이터 행 수를 센다(헤더 1줄 제외)."""
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_first_lines(path: Path) -> tuple[str, str]:
    with path.open(encoding="utf-8-sig") as handle:
        return handle.readline().strip(), handle.readline().strip()


def detect_delimiter(path: Path) -> str:
    """설계안 §4-2: 첫 데이터 행을 보고 ';' / '\\t' 중 구분자를 판별한다."""
    first, second = read_first_lines(path)
    if not first:
        raise ValueError(f"빈 파일: {path.name}")

    sample = second or first  # 데이터 행이 없으면 헤더로 판별
    sep = ";" if sample.count(";") == len(COLUMNS) - 1 else "\t"
    field_count = len(sample.split(sep))
    if field_count != len(COLUMNS):
        raise ValueError(
            f"구분자 판별 실패: {path.name} "
            f"(sep={sep!r}, 필드 {field_count}개, 기대 {len(COLUMNS)}개)"
        )
    return sep


def validate_header(path: Path, sep: str) -> None:
    """설계안 §4-3: 공식 11개 컬럼과 이름·순서가 같은지 확인한다."""
    first, _ = read_first_lines(path)
    header_cols = [col.strip(" \t;") for col in first.split(sep)]
    if header_cols != COLUMNS:
        raise ValueError(
            f"스키마 불일치: {path.name}\n"
            f"  기대({len(COLUMNS)}): {COLUMNS}\n"
            f"  실제({len(header_cols)}): {header_cols}"
        )


def iter_vm_chunks(path: Path, sep: str, chunksize: int):
    """설계안 §4-5: 청크 단위로 읽고 vm_id만 추가한다. 값은 변환하지 않는다."""
    vm_id = vm_id_of(path)
    reader = pd.read_csv(
        path,
        sep=sep,
        header=None,
        names=COLUMNS,
        skiprows=1,
        dtype=DTYPES,
        engine="c",
        encoding="utf-8-sig",
        on_bad_lines="error",
        chunksize=chunksize,
    )
    for chunk in reader:
        na_counts = chunk.isna().sum()          # 설계안 §4-6
        if na_counts.any():
            detail = ", ".join(f"{k}={v}" for k, v in na_counts[na_counts > 0].items())
            raise ValueError(f"결측/변환 실패 값 발견: {path.name} -> {detail}")
        chunk.insert(0, "vm_id", vm_id)
        yield chunk


def merge_trace(
    input_dir: Path,
    output_csv: Path,
    manifest_csv: Path,
    trace_name: str = TRACE_NAME,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    candidates = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=sort_key,
    )
    targets = [p for p in candidates if p.suffix.lower() in DATA_SUFFIXES]
    excluded = [p.name for p in candidates if p.suffix.lower() not in DATA_SUFFIXES]

    if not targets:
        raise SystemExit(f"데이터 파일을 찾지 못했습니다: {input_dir}")

    print(f"입력 디렉터리: {input_dir}")
    print(f"처리 대상: {len(targets):,}개  |  제외: {len(excluded)}개"
          + (f" ({', '.join(excluded[:5])}{' ...' if len(excluded) > 5 else ''})"
             if excluded else ""))
    print(f"청크 크기 : {chunksize:,}행")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(output_csv.suffix + ".tmp")
    if tmp_csv.exists():
        tmp_csv.unlink()

    records = []
    total_rows = 0
    total_chunks = 0
    header_written = False

    for index, path in enumerate(targets, 1):
        expected_rows = count_data_rows(path)
        sep = detect_delimiter(path)
        validate_header(path, sep)

        file_rows = 0
        for chunk in iter_vm_chunks(path, sep, chunksize):
            chunk.to_csv(
                tmp_csv,
                mode="a" if header_written else "w",
                index=False,
                header=not header_written,
                encoding="utf-8",
            )
            header_written = True
            file_rows += len(chunk)
            total_chunks += 1

        if file_rows != expected_rows:                      # 설계안 §4-7
            raise ValueError(
                f"행 수 불일치: {path.name} 원본 {expected_rows:,}행 != 적재 {file_rows:,}행"
            )

        total_rows += file_rows
        records.append({
            "trace": trace_name,
            "source_file": str(path.relative_to(input_dir.parent.parent)),
            "vm_id": vm_id_of(path),
            "rows": file_rows,
            "delimiter": DELIMITER_LABELS[sep],
            "loaded_file": str(output_csv),
        })

        if index % 100 == 0 or index == len(targets):
            print(f"  [{index:>5}/{len(targets)}] 누적 {total_rows:,}행 / 청크 {total_chunks:,}개")

    tmp_csv.replace(output_csv)                             # 설계안 §4-8

    manifest = pd.DataFrame(records)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8")

    source_rows = sum(count_data_rows(p) for p in targets)  # 전체 합계 재대조
    if source_rows != total_rows:
        raise ValueError(f"전체 행 수 불일치: 원본 {source_rows:,} != 적재 {total_rows:,}")

    print()
    print("[로딩 완료]")
    print(f"  출력      : {output_csv}  ({output_csv.stat().st_size / 1024**3:.2f} GiB)")
    print(f"  manifest  : {manifest_csv}")
    print(f"  VM 파일   : {len(records):,}개")
    print(f"  청크 수   : {total_chunks:,}개")
    print(f"  총 행 수  : {total_rows:,}행 (원본과 일치)")
    print(f"  컬럼 수   : {len(COLUMNS) + 1}개 (원본 {len(COLUMNS)} + vm_id)")
    print(f"  구분자    : {manifest['delimiter'].value_counts().to_dict()}")
    return manifest


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="GWA-T-12 트레이스 로딩")
    parser.add_argument("--trace", default=TRACE_NAME, help="트레이스 이름 (manifest 기록용)")
    parser.add_argument(
        "--input-dir", type=Path, default=root / "data" / TRACE_NAME / "2013-8",
        help="VM별 원본 파일이 있는 디렉터리",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data" / "vm_all_1250_loaded.csv",
        help="병합 결과 CSV 경로",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "data" / f"{TRACE_NAME}_loading_manifest.csv",
        help="manifest CSV 경로",
    )
    parser.add_argument(
        "--chunksize", type=int, default=CHUNKSIZE,
        help=f"read_csv 청크 크기 (기본 {CHUNKSIZE:,})",
    )
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        raise SystemExit(f"입력 디렉터리가 없습니다: {args.input_dir}")

    merge_trace(
        args.input_dir, args.output, args.manifest,
        trace_name=args.trace, chunksize=args.chunksize,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
