#!/usr/bin/env python3
"""GWA-T-12 정제 CSV 생성 스크립트.

전략: docs/GWA_T12_이상치_정제_전략.md (v1.3) §7.3·§7.4·§8
판정 기준선: docs/GWA_T12_정제_슬라이스B_동결기록.md

원칙
  - 판정은 다시 하지 않는다. 품질 로그(anomaly_events.csv)의 action=set_nan 이벤트만 적용한다
  - 동결본과 품질 로그는 읽기만 한다. 실행 전·후 SHA-256을 대조한다 (슬라이스 B 동결기록 §6)
  - 값 변경은 이벤트가 가리키는 셀만 한다: (vm_id, timestamp_raw, metric)이 같고
    값이 original_value와 같은 셀 (전략 §1-4 "해당 셀만", 중복 시각의 다른 값은 건드리지 않는다)
  - 행 수·행 순서·컬럼 12개·dtype은 동결본과 같다. 쓴 파일을 다시 읽어 값까지 대조한다
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clean_data import (
    CPU_CAP,
    CPU_USE,
    DTYPES,
    FROZEN_SHA256,
    IO_COLUMNS,
    MEM_CAP,
    MEM_USE,
    READ_COLUMNS,
    TS,
    sha256_of,
    verify_checksum,
    write_csv_atomic,
)

# 슬라이스 B 동결기록 §1: 판정 로그 기준선
FROZEN_EVENTS_SHA256 = "a6abfeff5bf50609913d03142fc856e5eb497abcbd310c11826f878bab662589"

VALUE_COLUMNS = READ_COLUMNS[2:]          # vm_id, Timestamp 제외 — set_nan 대상이 될 수 있는 컬럼
READ_OPTIONS = {"dtype": DTYPES, "float_precision": "round_trip"}   # 값 대조가 정확하도록


def load_set_nan_events(events_csv: Path) -> pd.DataFrame:
    events = pd.read_csv(
        events_csv,
        usecols=["vm_id", "timestamp_raw", "metric", "original_value", "action"],
        float_precision="round_trip",
    )
    events = events[events["action"] == "set_nan"].reset_index(drop=True)
    unknown = sorted(set(events["metric"]) - set(VALUE_COLUMNS))
    if unknown:
        raise ValueError(f"set_nan 이벤트의 metric이 데이터 컬럼이 아닙니다: {unknown}")
    return events


def apply_set_nan(df: pd.DataFrame, events: pd.DataFrame) -> int:
    """이벤트가 가리키는 셀을 결측으로 바꾸고 바뀐 셀 수를 돌려준다. 맞는 셀이 없는 이벤트는 오류."""
    if events.empty:
        return 0
    keys = pd.DataFrame({
        "vm_id": df["vm_id"].to_numpy(dtype=np.int64),
        "timestamp_raw": df[TS].to_numpy(dtype=np.int64),
        "row": np.arange(len(df)),
    })
    changed = 0
    for metric, ev in events.groupby("metric", sort=False):
        hits = ev.reset_index(names="event").merge(keys, on=["vm_id", "timestamp_raw"])
        current = df[metric].to_numpy(dtype="float64", na_value=np.nan)[hits["row"].to_numpy()]
        expected = hits["original_value"].to_numpy(dtype="float64")
        same = (current == expected) | (np.isnan(current) & np.isnan(expected))
        hits = hits[same]

        missing = sorted(set(ev.index) - set(hits["event"]))
        if missing:
            sample = events.loc[missing[:3], ["vm_id", "timestamp_raw", "metric", "original_value"]]
            raise ValueError(f"데이터에서 찾을 수 없는 set_nan 이벤트 {len(missing)}건:\n{sample}")

        rows = np.unique(hits["row"].to_numpy())
        df.iloc[rows, df.columns.get_loc(metric)] = np.nan   # Int64 컬럼은 <NA>가 된다
        changed += rows.size
    return changed


def peak_profile(df: pd.DataFrame) -> pd.Series:
    """전략 §8: 고부하가 훼손되지 않았는지 볼 지표 (최대 사용률·I/O 최고값·결측 수)."""
    cpu_cap = df[CPU_CAP].where(df[CPU_CAP] > 0)
    mem_cap = df[MEM_CAP].where(df[MEM_CAP] > 0)
    profile = {
        "rows": len(df),
        "max cpu_util_pct": (df[CPU_USE] / cpu_cap * 100).max(),
        "max memory_util_pct": (df[MEM_USE] / mem_cap * 100).max(),
        **{f"max {col}": df[col].max() for col in IO_COLUMNS},
        "missing cells": int(df[VALUE_COLUMNS].isna().sum().sum()),
    }
    return pd.Series(profile, dtype="float64")


def count_changed_cells(before: pd.DataFrame, after: pd.DataFrame) -> int:
    changed = 0
    for col in VALUE_COLUMNS:
        a = before[col].to_numpy(dtype="float64", na_value=np.nan)
        b = after[col].to_numpy(dtype="float64", na_value=np.nan)
        changed += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    return changed


def build_clean_csv(
    input_csv: Path,
    events_csv: Path,
    output_csv: Path,
    expected_sha256: str | None = FROZEN_SHA256,
    expected_events_sha256: str | None = FROZEN_EVENTS_SHA256,
) -> dict:
    if output_csv.resolve() in (input_csv.resolve(), events_csv.resolve()):
        raise ValueError(f"입력 파일을 덮어쓸 수 없습니다: {output_csv}")

    print(f"입력  : {input_csv}")
    print(f"이벤트: {events_csv}")
    verify_checksum(input_csv, expected_sha256, "동결본, 실행 전")
    verify_checksum(events_csv, expected_events_sha256, "판정 로그, 실행 전")

    df = pd.read_csv(input_csv, **READ_OPTIONS)
    if list(df.columns) != READ_COLUMNS:
        raise ValueError(f"컬럼 불일치\n  기대: {READ_COLUMNS}\n  실제: {list(df.columns)}")
    before = df.copy()
    print(f"  읽은 행    : {len(df):,}행 / 컬럼 {len(df.columns)}개")

    events = load_set_nan_events(events_csv)
    changed = apply_set_nan(df, events)
    print(f"  set_nan    : 이벤트 {len(events):,}건 → 셀 {changed:,}개 결측 처리")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(df, output_csv)

    # 쓴 파일을 다시 읽어 값·dtype·행 순서까지 대조한다
    written = pd.read_csv(output_csv, **READ_OPTIONS)
    pd.testing.assert_frame_equal(written, df)
    if count_changed_cells(before, written) != changed:
        raise ValueError("동결본 대비 바뀐 셀 수가 적용한 set_nan 수와 다릅니다")

    verify_checksum(input_csv, expected_sha256, "동결본, 실행 후")
    verify_checksum(events_csv, expected_events_sha256, "판정 로그, 실행 후")

    byte_identical = changed == 0 and sha256_of(output_csv) == sha256_of(input_csv)
    comparison = pd.DataFrame({"before": peak_profile(before), "after": peak_profile(written)})

    print()
    print("[정제 CSV 생성 완료]")
    print(f"  출력      : {output_csv}  ({output_csv.stat().st_size / 1024**3:.2f} GiB)")
    print(f"  행 수     : {len(written):,}행 (동결본과 일치) / 컬럼 {len(written.columns)}개")
    print(f"  수정 셀   : {changed:,}개")
    print(f"  바이트 비교: {'동결본과 바이트 단위로 동일' if byte_identical else '동결본과 다름 (수정 셀 반영 또는 표기 차이)'}")
    print("  정제 전·후 비교 (전략 §8):")
    for name, row in comparison.iterrows():
        mark = "" if row["before"] == row["after"] else "  ← 변화"
        print(f"    {name:<48} {row['before']:>18,.4f} {row['after']:>18,.4f}{mark}")
    return {"changed": changed, "rows": len(written), "byte_identical": byte_identical,
            "comparison": comparison, "frame": written}


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="GWA-T-12 정제 CSV 생성 (판정 로그의 set_nan만 적용)")
    parser.add_argument(
        "--input", type=Path, default=root / "data" / "vm_all_1250_loaded.csv",
        help="로딩 동결본 CSV (읽기 전용)",
    )
    parser.add_argument(
        "--events", type=Path, default=root / "data" / "quality" / "anomaly_events.csv",
        help="판정 이벤트 로그 (읽기 전용)",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data" / "vm_all_1250_clean.csv",
        help="정제 CSV 경로",
    )
    parser.add_argument(
        "--expected-sha256", default=FROZEN_SHA256,
        help="동결본 체크섬 기준 (로딩 동결기록 §1). 빈 문자열이면 대조를 생략한다",
    )
    parser.add_argument(
        "--expected-events-sha256", default=FROZEN_EVENTS_SHA256,
        help="판정 로그 체크섬 기준 (슬라이스 B 동결기록 §1). 빈 문자열이면 대조를 생략한다",
    )
    args = parser.parse_args(argv)

    for path in (args.input, args.events):
        if not path.is_file():
            raise SystemExit(f"입력 파일이 없습니다: {path}")

    build_clean_csv(
        args.input, args.events, args.output,
        expected_sha256=args.expected_sha256 or None,
        expected_events_sha256=args.expected_events_sha256 or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
