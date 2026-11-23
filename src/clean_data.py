#!/usr/bin/env python3
"""GWA-T-12 정제 판정 스크립트 — 슬라이스 A (공통 시간 검사 + CPU/Memory) + B (Disk/Network).

전략: docs/GWA_T12_이상치_정제_전략.md (v1.3)

원칙
  - 동결본(data/vm_all_1250_loaded.csv)은 읽기만 한다. 실행 전·후 SHA-256을 대조한다
  - 품질 로그만 만든다. 정제 CSV는 별도 단계에서 생성한다 (§7.4)
  - 값 변경 대상은 INVALID뿐이며(action=set_nan), SUSPECT·UNDETERMINED는 값을 유지한다
  - 할당 0은 오류가 아닌 상태로 보고 구간만 기록한다 (§3.1)
  - 중복 시각은 판정용 시계열에서만 제거하고, 이벤트는 (vm_id, timestamp_raw, metric) 키로 남긴다 (§3.2)
  - Group B 행은 VM마다 Group A 행 뒤에 붙인다 (슬라이스 A 동결기록 §6: A 부분 체크섬 유지)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TS = "Timestamp [ms]"
CORES = "CPU cores"
CPU_CAP = "CPU capacity provisioned [MHZ]"
CPU_USE = "CPU usage [MHZ]"
CPU_PCT = "CPU usage [%]"
MEM_CAP = "Memory capacity provisioned [KB]"
MEM_USE = "Memory usage [KB]"

IO_COLUMNS = [
    "Disk read throughput [KB/s]",
    "Disk write throughput [KB/s]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

GROUP_A_COLUMNS = ["vm_id", TS, CORES, CPU_CAP, CPU_USE, CPU_PCT, MEM_CAP, MEM_USE]
READ_COLUMNS = GROUP_A_COLUMNS + IO_COLUMNS
VALIDITY_COLUMNS = [CORES, CPU_CAP, CPU_USE, CPU_PCT, MEM_CAP, MEM_USE]
DTYPES = {TS: "Int64", CORES: "Int64"}          # 동결기록 §2

# 동결기록 §1: 로딩 동결본 기준선
FROZEN_SHA256 = "ee2426537153c0dd024e8a828a94acf6984f9940353abdc253574b4a93e65072"

# 전략 §4.2 설정값 v1
WINDOW = 13
MIN_VALID = 7
THRESHOLD = 3.0
EXPECTED_STEP = 300      # 초 (동결기록 §5-3)
SEGMENT_GAP = 600        # 간격이 이보다 크면 새 구간
PCT_TOLERANCE = 0.01     # 전략 §3: CPU usage [%] 일치 허용 오차 (%p)

# 전략 §5.2 설정값 v1 (log1p 변환, 상방 단측)
IO_WINDOW = 25
IO_MIN_VALID = 12
IO_THRESHOLD = 6.0

# 자원별 (사용량 컬럼, 할당 컬럼). 판정 이벤트의 metric은 사용량 컬럼명 (§7.3)
RESOURCES = {
    "cpu": (CPU_USE, CPU_CAP),
    "memory": (MEM_USE, MEM_CAP),
}

EVENT_COLUMNS = ["vm_id", "timestamp_raw", "metric", "group", "status", "reason",
                 "original_value", "action"]
SUMMARY_COLUMNS = ["vm_id", "metric", "group", "status", "reason", "count"]
RUN_COLUMNS = ["vm_id", "resource", "start_timestamp_raw", "end_timestamp_raw", "rows",
               "position", "rows_usage_positive"]

EVENTS_NAME = "anomaly_events.csv"
SUMMARY_NAME = "anomaly_summary.csv"
RUNS_NAME = "zero_allocation_runs.csv"

# Hampel 판정 코드
NOT_EVALUATED, INSUFFICIENT, ZERO_MAD, NORMAL, SPIKE = range(5)


def sha256_of(path: Path, block: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str | None, stage: str) -> None:
    """전략 §8: 동결본이 동결기록 §1과 같은지 대조한다. expected가 None이면 생략."""
    if expected is None:
        return
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"동결본 체크섬 불일치({stage}): {path}\n  기대: {expected}\n  실제: {actual}")
    print(f"  체크섬 일치({stage}): {actual[:16]}…")


def segment_ids(ts, gap: int = SEGMENT_GAP) -> np.ndarray:
    """전략 §4.2-6: 인접 간격이 gap초를 넘으면 새 구간 번호를 붙인다."""
    ts = np.asarray(ts, dtype=np.int64)
    return np.cumsum(np.diff(ts, prepend=ts[:1]) > gap)


def hampel_verdict(x, segment, window: int = WINDOW, min_valid: int = MIN_VALID,
                   threshold: float = THRESHOLD, upper_only: bool = False) -> np.ndarray:
    """전략 §4.2·§5.2: 직전 window개 시점(같은 구간)의 유효값으로 중앙값·MAD를 구해 판정한다.

    NaN(할당 0·INVALID·duplicate_conflict)은 이력에서 건너뛰고, x가 NaN인 시점은 NOT_EVALUATED.
    upper_only=True면 중앙값보다 위로 벗어난 경우만 급증으로 본다 (Group B).
    """
    x = np.asarray(x, dtype=float)
    segment = np.asarray(segment)
    verdict = np.full(x.size, NOT_EVALUATED, dtype=np.int8)
    if x.size == 0:
        return verdict

    # 시점 i의 이력 = x[i-window : i]  (shift(1)과 같은 효과로 현재값은 섞이지 않는다)
    hist = sliding_window_view(np.concatenate([np.full(window, np.nan), x[:-1]]), window).copy()
    hist_seg = sliding_window_view(np.concatenate([np.full(window, -1), segment[:-1]]), window)
    hist[hist_seg != segment[:, None]] = np.nan

    n_valid = np.count_nonzero(~np.isnan(hist), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 유효값이 하나도 없는 창
        median = np.nanmedian(hist, axis=1)
        mad = np.nanmedian(np.abs(hist - median[:, None]), axis=1)
    sigma = 1.4826 * mad

    valid = ~np.isnan(x)
    enough = valid & (n_valid >= min_valid)
    judged = enough & (sigma > 0)
    deviation = (x - median) if upper_only else np.abs(x - median)
    spike = judged & (deviation > threshold * sigma)
    verdict[valid & ~enough] = INSUFFICIENT
    verdict[enough & (sigma == 0)] = ZERO_MAD          # MAD=0이면 임계값을 임의로 만들지 않는다
    verdict[judged & ~spike] = NORMAL
    verdict[spike] = SPIKE
    return verdict


def zero_allocation_runs(zero, ts, usage) -> list[dict]:
    """전략 §3.1: 할당 0이 이어지는 구간을 파일 행 기준(중복 행 포함)으로 찾는다."""
    zero = np.asarray(zero, dtype=bool)
    if not zero.any():
        return []
    edges = np.diff(np.concatenate([[0], zero.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    last = zero.size - 1
    runs = []
    for start, end in zip(starts, ends):
        if start == 0:
            position = "whole" if end == last else "leading"
        else:
            position = "trailing" if end == last else "middle"
        runs.append({
            "start_timestamp_raw": int(ts[start]),
            "end_timestamp_raw": int(ts[end]),
            "rows": int(end - start + 1),
            "position": position,
            "rows_usage_positive": int((usage[start:end + 1] > 0).sum()),
        })
    return runs


def _events(vm_id, ts, values, mask, metric, group, status, reason) -> pd.DataFrame | None:
    """mask가 True인 행을 이벤트로 만든다. 같은 (시각, 값)은 한 줄로 합친다 (§7.3)."""
    if not mask.any():
        return None
    frame = pd.DataFrame({
        "vm_id": vm_id,
        "timestamp_raw": ts[mask],
        "metric": metric,
        "group": group,
        "status": status,
        "reason": reason,
        "original_value": values[mask],
        "action": "set_nan" if status == "INVALID" else "keep",
    })
    return frame.drop_duplicates(["timestamp_raw", "original_value"])


def check_vm(vm_id, g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """한 VM(파일 행 순서)에 슬라이스 A·B 검사를 적용해 (events, summary, runs)를 돌려준다."""
    ts = g[TS].to_numpy(dtype=np.int64)
    step = np.diff(ts)
    if (step < 0).any():
        raise ValueError(f"vm {vm_id}: timestamp 역행 발견 — 전략 §4.2-1 전제(VM 내부 시간순) 위반")

    first = np.concatenate([[True], step != 0])   # 시각별 첫 행 (중복은 인접해 있다)
    grp = np.cumsum(first) - 1                     # 각 행이 속한 시점 번호
    n_ts = int(first.sum())
    ts_first = ts[first]
    seg = segment_ids(ts_first)

    def per_ts_any(mask):
        return np.bincount(grp, weights=mask, minlength=n_ts) > 0

    events, summary, runs = [], [], []

    def tally(metric, group, status, reason, count):
        summary.append({"vm_id": vm_id, "metric": metric, "group": group,
                        "status": status, "reason": reason, "count": int(count)})

    # --- 공통: 시간 간격 (전략 §3, 인접 간격 단위) ---
    tally(TS, "common", "VALID", "regular_step", (step == EXPECTED_STEP).sum())
    tally(TS, "common", "UNDETERMINED", "duplicate_timestamp", (step == 0).sum())
    tally(TS, "common", "UNDETERMINED", "irregular_step",
          ((step > 0) & (step != EXPECTED_STEP) & (step <= SEGMENT_GAP)).sum())
    tally(TS, "common", "UNDETERMINED", "segment_break", (step > SEGMENT_GAP).sum())

    # --- 공통 1차 검사: 음수·비유한 값만 INVALID (전략 §3) ---
    values = {col: g[col].to_numpy(dtype="float64", na_value=np.nan) for col in VALIDITY_COLUMNS}
    bad = {}
    for col in VALIDITY_COLUMNS:
        v = values[col]
        non_finite = ~np.isfinite(v)
        negative = v < 0
        bad[col] = non_finite | negative
        for reason, mask in (("negative_value", negative), ("non_finite", non_finite)):
            tally(col, "resource", "INVALID", reason, per_ts_any(mask).sum())
            events.append(_events(vm_id, ts, v, mask, col, "resource", "INVALID", reason))

    cores_zero = values[CORES] == 0

    # --- 자원별: 할당 0 구간 · 할당량 초과 · Hampel (전략 §3.1, §3.2, §4.2) ---
    for resource, (use_col, cap_col) in RESOURCES.items():
        use, cap = values[use_col], values[cap_col]
        zero = cap == 0
        if resource == "cpu":
            zero |= cores_zero
        usable = ~zero & ~bad[use_col] & ~bad[cap_col]

        for run in zero_allocation_runs(zero, ts, use):
            runs.append({"vm_id": vm_id, "resource": resource, **run})

        over = usable & (use > cap)
        tally(use_col, "resource", "SUSPECT", "over_capacity", per_ts_any(over).sum())
        events.append(_events(vm_id, ts, use, over, use_col, "resource", "SUSPECT", "over_capacity"))

        util = np.full(use.size, np.nan)
        util[usable] = use[usable] / cap[usable] * 100

        # 같은 시각 안에서 사용률이 다르면 충돌 (NaN끼리는 같은 값으로 본다)
        ref = util[first][grp]
        differs = ~((util == ref) | (np.isnan(util) & np.isnan(ref)))
        conflict_ts = per_ts_any(differs)
        tally(use_col, "resource", "UNDETERMINED", "duplicate_conflict", conflict_ts.sum())
        events.append(_events(vm_id, ts, use, conflict_ts[grp], use_col, "resource",
                              "UNDETERMINED", "duplicate_conflict"))

        zero_ts = zero[first] & ~conflict_ts
        tally(use_col, "resource", "UNDETERMINED", "zero_allocation", zero_ts.sum())

        x = util[first].copy()
        x[conflict_ts] = np.nan
        verdict = hampel_verdict(x, seg)
        tally(use_col, "resource", "VALID", "hampel_normal", (verdict == NORMAL).sum())
        tally(use_col, "resource", "SUSPECT", "hampel_spike", (verdict == SPIKE).sum())
        tally(use_col, "resource", "UNDETERMINED", "insufficient_history", (verdict == INSUFFICIENT).sum())
        tally(use_col, "resource", "UNDETERMINED", "zero_mad", (verdict == ZERO_MAD).sum())
        events.append(_events(vm_id, ts_first, use[first], verdict == SPIKE, use_col, "resource",
                              "SUSPECT", "hampel_spike"))

    # --- CPU usage [%] 일치 검사 (전략 §3 스키마 중복 유의) ---
    pct = values[CPU_PCT]
    checked = ~cores_zero & (values[CPU_CAP] > 0) & ~bad[CPU_USE] & ~bad[CPU_CAP] & ~bad[CPU_PCT]
    expected_pct = np.full(pct.size, np.nan)
    expected_pct[checked] = values[CPU_USE][checked] / values[CPU_CAP][checked] * 100
    mismatch = checked & (np.abs(pct - expected_pct) > PCT_TOLERANCE)
    mismatch_ts = per_ts_any(mismatch)
    tally(CPU_PCT, "resource", "VALID", "pct_match", (per_ts_any(checked) & ~mismatch_ts).sum())
    tally(CPU_PCT, "resource", "SUSPECT", "pct_mismatch", mismatch_ts.sum())
    events.append(_events(vm_id, ts, pct, mismatch, CPU_PCT, "resource", "SUSPECT", "pct_mismatch"))

    # --- Group B: Disk/Network 처리량 (전략 §5). 반드시 Group A 행 뒤에 붙인다 ---
    for col in IO_COLUMNS:
        v = g[col].to_numpy(dtype="float64", na_value=np.nan)
        non_finite = ~np.isfinite(v)
        negative = v < 0                               # 0은 유효 (§5.2 1차)
        for reason, mask in (("negative_value", negative), ("non_finite", non_finite)):
            tally(col, "io", "INVALID", reason, per_ts_any(mask).sum())
            events.append(_events(vm_id, ts, v, mask, col, "io", "INVALID", reason))

        y = np.full(v.size, np.nan)
        valid = ~(non_finite | negative)
        y[valid] = np.log1p(v[valid])                  # 판정용 변환. 원본 v는 바꾸지 않는다

        ref = y[first][grp]
        differs = ~((y == ref) | (np.isnan(y) & np.isnan(ref)))
        conflict_ts = per_ts_any(differs)
        tally(col, "io", "UNDETERMINED", "duplicate_conflict", conflict_ts.sum())
        events.append(_events(vm_id, ts, v, conflict_ts[grp], col, "io",
                              "UNDETERMINED", "duplicate_conflict"))

        x = y[first].copy()
        x[conflict_ts] = np.nan
        verdict = hampel_verdict(x, seg, IO_WINDOW, IO_MIN_VALID, IO_THRESHOLD, upper_only=True)
        tally(col, "io", "VALID", "hampel_normal", (verdict == NORMAL).sum())
        tally(col, "io", "SUSPECT", "hampel_spike", (verdict == SPIKE).sum())
        tally(col, "io", "UNDETERMINED", "insufficient_history", (verdict == INSUFFICIENT).sum())
        tally(col, "io", "UNDETERMINED", "zero_mad", (verdict == ZERO_MAD).sum())
        events.append(_events(vm_id, ts_first, v[first], verdict == SPIKE, col, "io",
                              "SUSPECT", "hampel_spike"))

    events_df = concat_frames([e for e in events if e is not None], EVENT_COLUMNS)
    return (events_df[EVENT_COLUMNS],
            pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
            pd.DataFrame(runs, columns=RUN_COLUMNS))


def concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """빈 프레임을 빼고 이어 붙인다 (빈 프레임 concat은 pandas가 dtype 경고를 낸다)."""
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def run(input_csv: Path, quality_dir: Path, expected_sha256: str | None = FROZEN_SHA256) -> dict:
    print(f"입력: {input_csv}")
    verify_checksum(input_csv, expected_sha256, "실행 전")

    df = pd.read_csv(input_csv, usecols=READ_COLUMNS, dtype=DTYPES)
    vm = df["vm_id"].to_numpy()
    if (np.diff(vm) < 0).any():                    # 전략 §7.1: vm_id 연속 구간 전제
        raise ValueError("vm_id가 연속 구간으로 저장되어 있지 않습니다 (전략 §7.1 전제 위반)")
    print(f"  읽은 행  : {len(df):,}행 / VM {df['vm_id'].nunique():,}개")

    events, summary, runs = [], [], []
    for index, (vm_id, g) in enumerate(df.groupby("vm_id", sort=False), 1):
        e, s, r = check_vm(vm_id, g)
        events.append(e)
        summary.append(s)
        runs.append(r)
        if index % 250 == 0:
            print(f"  [{index:>5}] VM 판정 완료")

    events_df = concat_frames(events, EVENT_COLUMNS)
    summary_df = concat_frames(summary, SUMMARY_COLUMNS)
    runs_df = concat_frames(runs, RUN_COLUMNS)

    quality_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "events": quality_dir / EVENTS_NAME,
        "summary": quality_dir / SUMMARY_NAME,
        "runs": quality_dir / RUNS_NAME,
    }
    write_csv_atomic(events_df, outputs["events"])
    write_csv_atomic(summary_df, outputs["summary"])
    write_csv_atomic(runs_df, outputs["runs"])

    verify_checksum(input_csv, expected_sha256, "실행 후")

    totals = summary_df.groupby(["metric", "status", "reason"], sort=False)["count"].sum()
    print()
    print("[슬라이스 A·B 판정 완료]")
    for name, path in outputs.items():
        print(f"  {name:<8}: {path}")
    print(f"  이벤트   : {len(events_df):,}행  {events_df['status'].value_counts().to_dict()}")
    print(f"  할당 0 구간: {len(runs_df):,}개  {runs_df['resource'].value_counts().to_dict()}")
    print("  집계 (전체 VM 합계):")
    for (metric, status, reason), count in totals.items():
        print(f"    {metric:<34} {status:<13} {reason:<22} {count:>12,}")
    return {"events": events_df, "summary": summary_df, "runs": runs_df, "outputs": outputs}


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="GWA-T-12 정제 판정 (슬라이스 A: 공통 시간 + CPU/Memory, B: Disk/Network)"
    )
    parser.add_argument(
        "--input", type=Path, default=root / "data" / "vm_all_1250_loaded.csv",
        help="로딩 동결본 CSV (읽기 전용)",
    )
    parser.add_argument(
        "--quality-dir", type=Path, default=root / "data" / "quality",
        help="품질 로그 출력 디렉터리",
    )
    parser.add_argument(
        "--expected-sha256", default=FROZEN_SHA256,
        help="입력 체크섬 기준 (동결기록 §1). 빈 문자열이면 대조를 생략한다",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"입력 파일이 없습니다: {args.input}")

    run(args.input, args.quality_dir, expected_sha256=args.expected_sha256 or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
