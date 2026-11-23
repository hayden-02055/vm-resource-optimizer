"""src/clean_data.py 슬라이스 A·B 테스트 — 전략 문서 v1.3 규칙을 합성 데이터로 검증한다."""
import numpy as np
import pandas as pd
import pytest

import clean_data as cd
from clean_data import (
    CORES,
    CPU_CAP,
    CPU_PCT,
    CPU_USE,
    INSUFFICIENT,
    IO_COLUMNS,
    MEM_CAP,
    MEM_USE,
    NORMAL,
    NOT_EVALUATED,
    SPIKE,
    TS,
    ZERO_MAD,
)

# 사용률(%) 기준 잡음 있는 평상시 패턴: 직전 13개 중앙값 ≈ 21, MAD = 1
BASE = [20, 21, 20, 22, 21, 20, 21, 22, 20, 21, 20, 21, 22, 21]

# 처리량(KB/s) 평상시 패턴: log1p 공간에서 6σ 상한 ≈ 24 KB/s
DISK_R, DISK_W, NET_RX, NET_TX = IO_COLUMNS
IO_BASE = [10.0, 12.0, 11.0, 13.0] * 7


def make_vm(ts, cpu_use, cpu_cap=1000.0, mem_use=None, mem_cap=1000.0, cores=2,
            pct=None, vm_id=1, io=None):
    n = len(ts)
    io = io or {}

    def full(value, dtype=float):
        return np.broadcast_to(np.asarray(value, dtype=dtype), (n,)).copy()

    cpu_use = np.asarray(cpu_use, dtype=float)
    cpu_cap = full(cpu_cap)
    if pct is None:
        pct = np.where(cpu_cap > 0, cpu_use / np.where(cpu_cap > 0, cpu_cap, 1) * 100, 0.0)
    return pd.DataFrame({
        "vm_id": vm_id,
        TS: pd.array(ts, dtype="Int64"),
        CORES: pd.array(full(cores, int), dtype="Int64"),
        CPU_CAP: cpu_cap,
        CPU_USE: cpu_use,
        CPU_PCT: np.asarray(pct, dtype=float),
        MEM_CAP: full(mem_cap),
        MEM_USE: full(500.0) if mem_use is None else np.asarray(mem_use, dtype=float),
        **{col: np.asarray(io.get(col, np.zeros(n)), dtype=float) for col in IO_COLUMNS},
    })


def count(summary, metric, reason):
    row = summary[(summary["metric"] == metric) & (summary["reason"] == reason)]
    assert len(row) == 1, f"집계 행이 정확히 1개여야 함: {metric} / {reason}"
    return int(row["count"].iloc[0])


def hampel_total(summary, metric):
    return sum(count(summary, metric, r)
               for r in ("hampel_normal", "hampel_spike", "insufficient_history", "zero_mad"))


def events_of(events, metric, reason):
    return events[(events["metric"] == metric) & (events["reason"] == reason)]


# --- hampel_verdict ---------------------------------------------------------

def test_hampel_flags_spikes_in_both_directions():
    x = BASE + [95] + BASE[:5] + [1]
    v = cd.hampel_verdict(x, np.zeros(len(x), dtype=int))
    assert v[14] == SPIKE and v[20] == SPIKE
    assert (v[7:14] == NORMAL).all() and (v[15:20] == NORMAL).all()


def test_hampel_upper_only_ignores_drops():
    x = BASE + [95] + BASE[:5] + [1]
    v = cd.hampel_verdict(x, np.zeros(len(x), dtype=int), upper_only=True)
    assert v[14] == SPIKE
    assert v[20] == NORMAL


def test_hampel_insufficient_history_until_min_valid():
    v = cd.hampel_verdict(BASE, np.zeros(len(BASE), dtype=int))
    assert (v[:cd.MIN_VALID] == INSUFFICIENT).all()
    assert v[cd.MIN_VALID] == NORMAL


def test_hampel_zero_mad_is_not_judged_even_for_jump():
    x = [50.0] * 10 + [99.0]
    v = cd.hampel_verdict(x, np.zeros(len(x), dtype=int))
    assert (v[7:] == ZERO_MAD).all()


def test_hampel_skips_nan_in_history():
    """회귀: v1.1 예시(np.median)는 창에 NaN이 하나라도 있으면 MAD가 NaN이 되어 이 급증을 놓쳤다."""
    x = np.array(BASE + [95], dtype=float)
    x[10] = np.nan
    v = cd.hampel_verdict(x, np.zeros(len(x), dtype=int))
    assert v[10] == NOT_EVALUATED
    assert (v[11:14] == NORMAL).all()
    assert v[14] == SPIKE


def test_hampel_window_counts_time_points_not_valid_values():
    seg = np.zeros(14, dtype=int)
    seven_valid = BASE[:7] + [np.nan] * 6 + [21]
    six_valid = BASE[:6] + [np.nan] * 7 + [21]
    assert cd.hampel_verdict(seven_valid, seg)[13] == NORMAL
    assert cd.hampel_verdict(six_valid, seg)[13] == INSUFFICIENT


def test_hampel_segment_break_resets_history():
    x = BASE + BASE
    seg = np.array([0] * 14 + [1] * 14)
    v = cd.hampel_verdict(x, seg)
    assert (v[14:14 + cd.MIN_VALID] == INSUFFICIENT).all()
    assert v[14 + cd.MIN_VALID] == NORMAL


def test_segment_ids_break_only_above_gap():
    ts = [0, 300, 600, 1200, 1500, 2101, 2401]   # 간격 600은 유지, 601은 분리
    assert cd.segment_ids(ts).tolist() == [0, 0, 0, 0, 0, 1, 1]


# --- zero_allocation_runs ---------------------------------------------------

def test_zero_allocation_runs_positions_and_usage():
    ts = np.arange(7) * 300
    zero = np.array([0, 1, 1, 0, 0, 1, 1], dtype=bool)
    usage = np.array([0, 5, 0, 0, 0, 0, 3], dtype=float)
    runs = cd.zero_allocation_runs(zero, ts, usage)
    assert [(r["start_timestamp_raw"], r["end_timestamp_raw"], r["rows"], r["position"],
             r["rows_usage_positive"]) for r in runs] == [
        (300, 600, 2, "middle", 1),
        (1500, 1800, 2, "trailing", 1),
    ]
    two = np.arange(2)
    assert cd.zero_allocation_runs([True, False], two, two)[0]["position"] == "leading"
    assert cd.zero_allocation_runs([True, True], two, two)[0]["position"] == "whole"
    assert cd.zero_allocation_runs([False, False], two, two) == []


# --- check_vm ---------------------------------------------------------------

def test_zero_allocation_is_kept_as_state_not_invalid():
    n_live, n_zero = 15, 5
    ts = np.arange(n_live + n_zero) * 300
    g = make_vm(
        ts,
        cpu_use=[u * 10 for u in BASE] + [210] + [10, 0, 0, 0, 0],   # 할당 0인데 사용량 10
        cpu_cap=[1000.0] * n_live + [0.0] * n_zero,
        mem_use=[500.0] * n_live + [0.0] * n_zero,
        mem_cap=[1000.0] * n_live + [0.0] * n_zero,
        cores=[2] * n_live + [0] * n_zero,
    )
    events, summary, runs = cd.check_vm(1, g)

    assert (events["status"] != "INVALID").all()
    assert events_of(events, CPU_USE, "over_capacity").empty
    assert count(summary, CPU_USE, "zero_allocation") == n_zero
    assert count(summary, MEM_USE, "zero_allocation") == n_zero
    assert hampel_total(summary, CPU_USE) == n_live
    assert runs[["resource", "rows", "position", "rows_usage_positive"]].values.tolist() == [
        ["cpu", n_zero, "trailing", 1],
        ["memory", n_zero, "trailing", 0],
    ]


def test_duplicates_are_judged_once_and_counted():
    util = BASE + [95, 21]
    ts = np.repeat(np.arange(len(util)) * 300, 2)              # 모든 시각이 2행씩
    g = make_vm(ts, cpu_use=np.repeat([u * 10 for u in util], 2))
    events, summary, _ = cd.check_vm(1, g)

    spikes = events_of(events, CPU_USE, "hampel_spike")
    assert spikes["timestamp_raw"].tolist() == [14 * 300]
    assert count(summary, TS, "duplicate_timestamp") == len(util)
    assert count(summary, CPU_USE, "duplicate_conflict") == 0
    assert hampel_total(summary, CPU_USE) == len(util)        # 행 수(32)가 아니라 시점 수(16)


def test_duplicate_conflict_is_logged_and_excluded_from_history():
    util = BASE + [21, 21]
    ts = list(np.arange(len(util)) * 300)
    cpu = [u * 10.0 for u in util]
    ts.insert(6, ts[5])                                         # 시각 1500에 값이 다른 중복 행
    cpu.insert(6, 900.0)
    g = make_vm(ts, cpu_use=cpu)
    events, summary, _ = cd.check_vm(1, g)

    conflict = events_of(events, CPU_USE, "duplicate_conflict")
    assert sorted(conflict["original_value"].tolist()) == [200.0, 900.0]
    assert (conflict["timestamp_raw"] == 1500).all() and (conflict["action"] == "keep").all()
    assert count(summary, CPU_USE, "duplicate_conflict") == 1
    assert count(summary, MEM_USE, "duplicate_conflict") == 0  # 메모리 값은 같으므로 충돌 아님
    assert hampel_total(summary, CPU_USE) == len(util) - 1


def test_negative_and_non_finite_are_invalid_set_nan():
    cpu = [u * 10.0 for u in BASE]
    mem = [500.0] * len(BASE)
    cpu[3] = -50.0
    mem[5] = np.inf
    g = make_vm(np.arange(len(BASE)) * 300, cpu_use=cpu, mem_use=mem)
    events, summary, _ = cd.check_vm(1, g)

    neg = events_of(events, CPU_USE, "negative_value")
    inf = events_of(events, MEM_USE, "non_finite")
    assert neg[["timestamp_raw", "status", "action"]].values.tolist() == [[900, "INVALID", "set_nan"]]
    assert inf[["timestamp_raw", "status", "action"]].values.tolist() == [[1500, "INVALID", "set_nan"]]
    assert hampel_total(summary, CPU_USE) == len(BASE) - 1    # INVALID 시점은 판정에서 제외
    assert count(summary, CORES, "negative_value") == 0        # 0건 검사도 집계에 남는다


def test_over_capacity_and_pct_mismatch_are_suspect_keep():
    util = BASE + [21, 120, 21, 21]
    cpu = [u * 10.0 for u in util]
    pct = [u * 1.0 for u in util]
    pct[16] += 0.5                                             # 허용 오차 초과
    pct[17] += 0.005                                           # 허용 오차 이내
    g = make_vm(np.arange(len(util)) * 300, cpu_use=cpu, pct=pct)
    events, summary, _ = cd.check_vm(1, g)

    over = events_of(events, CPU_USE, "over_capacity")
    mism = events_of(events, CPU_PCT, "pct_mismatch")
    assert over[["timestamp_raw", "status", "action"]].values.tolist() == [[15 * 300, "SUSPECT", "keep"]]
    assert mism["timestamp_raw"].tolist() == [16 * 300]
    assert count(summary, CPU_PCT, "pct_match") == len(util) - 1


def test_timestamp_tally_partitions_steps():
    ts = [0, 300, 300, 600, 650, 950, 2000, 2300]  # 300, 0, 300, 50, 300, 1050, 300
    g = make_vm(ts, cpu_use=[200.0] * len(ts))
    _, summary, _ = cd.check_vm(1, g)
    tallies = [count(summary, TS, r) for r in
               ("regular_step", "duplicate_timestamp", "irregular_step", "segment_break")]
    assert tallies == [4, 1, 1, 1]
    assert sum(tallies) == len(ts) - 1


def test_backward_timestamp_raises():
    g = make_vm([0, 300, 200], cpu_use=[200.0] * 3)
    with pytest.raises(ValueError, match="역행"):
        cd.check_vm(1, g)


# --- check_vm: Group B (Disk/Network) ----------------------------------------

def test_io_spike_is_upper_only_and_logs_raw_value():
    disk_w = IO_BASE[:26] + [5000.0, 11.0, 0.0, 12.0]          # 26: 급증, 28: 급락(0)
    n = len(disk_w)
    g = make_vm(np.arange(n) * 300, cpu_use=[200.0] * n, io={DISK_W: disk_w})
    events, summary, _ = cd.check_vm(1, g)

    spikes = events_of(events, DISK_W, "hampel_spike")
    assert spikes[["timestamp_raw", "group", "status", "original_value", "action"]].values.tolist() == [
        [26 * 300, "io", "SUSPECT", 5000.0, "keep"]             # log1p 값이 아니라 원래 처리량
    ]
    assert count(summary, DISK_W, "insufficient_history") == cd.IO_MIN_VALID
    assert hampel_total(summary, DISK_W) == n


def test_io_zero_run_then_first_positive_is_kept_unjudged():
    """전략 §5.2-4: 0이 이어지다가 나타난 첫 양수 값은 MAD=0이라 판정하지 않는다."""
    disk_r = [0.0] * 20 + [800.0] + [0.0] * 5
    n = len(disk_r)
    g = make_vm(np.arange(n) * 300, cpu_use=[200.0] * n, io={DISK_R: disk_r})
    events, summary, _ = cd.check_vm(1, g)

    assert (events["group"] != "io").all()                      # 0은 INVALID 아님, 급증 후보도 없음
    assert count(summary, DISK_R, "zero_mad") == n - cd.IO_MIN_VALID
    assert count(summary, DISK_R, "negative_value") == 0


def test_io_negative_and_non_finite_are_invalid_set_nan():
    n = len(IO_BASE)
    net_rx, net_tx = list(IO_BASE), list(IO_BASE)
    net_rx[3] = -1.0
    net_tx[5] = np.inf
    g = make_vm(np.arange(n) * 300, cpu_use=[200.0] * n, io={NET_RX: net_rx, NET_TX: net_tx})
    events, summary, _ = cd.check_vm(1, g)

    neg = events_of(events, NET_RX, "negative_value")
    inf = events_of(events, NET_TX, "non_finite")
    assert neg[["timestamp_raw", "group", "action"]].values.tolist() == [[900, "io", "set_nan"]]
    assert inf[["timestamp_raw", "group", "action"]].values.tolist() == [[1500, "io", "set_nan"]]
    assert hampel_total(summary, NET_RX) == n - 1


def test_io_duplicate_conflict_is_logged_per_metric():
    ts = list(np.arange(20) * 300)
    disk_w = IO_BASE[:20]
    ts.insert(6, ts[5])                                         # 시각 1500에 Disk write만 다른 중복 행
    disk_w.insert(6, 999.0)
    g = make_vm(ts, cpu_use=[200.0] * len(ts), io={DISK_W: disk_w})
    events, summary, _ = cd.check_vm(1, g)

    conflict = events_of(events, DISK_W, "duplicate_conflict")
    assert sorted(conflict["original_value"].tolist()) == [12.0, 999.0]
    assert count(summary, DISK_W, "duplicate_conflict") == 1
    assert count(summary, DISK_R, "duplicate_conflict") == 0
    assert count(summary, CPU_USE, "duplicate_conflict") == 0


def test_group_a_output_is_independent_of_io_and_comes_first():
    """슬라이스 A 동결기록 §6: B를 붙여도 Group A 행 내용·순서는 그대로, VM마다 A 행이 B 행보다 앞."""
    util = BASE + [95, 21]
    n = len(util)
    ts = np.arange(n) * 300
    quiet = make_vm(ts, cpu_use=[u * 10 for u in util])
    busy = make_vm(ts, cpu_use=[u * 10 for u in util],
                   io={DISK_W: IO_BASE[:14] + [5000.0, 11.0], NET_RX: np.arange(n) * 7.0})
    eq, sq, _ = cd.check_vm(1, quiet)
    eb, sb, _ = cd.check_vm(1, busy)

    def group_a(frame):
        return frame[frame["group"] != "io"].reset_index(drop=True)

    pd.testing.assert_frame_equal(group_a(eq), group_a(eb))
    pd.testing.assert_frame_equal(group_a(sq), group_a(sb))
    for frame in (eb, sb):
        groups = frame["group"].tolist()
        first_io = groups.index("io")
        assert set(groups[first_io:]) == {"io"} and "io" not in groups[:first_io]
    assert len(sb) == 60 and (sb["group"] != "io").sum() == 32


# --- run (파일 입출력) -------------------------------------------------------

def write_loaded_csv(path, frames):
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)   # 동결본과 같은 12컬럼 형태


def two_vm_frames():
    ts = np.arange(len(BASE) + 1) * 300
    vm1 = make_vm(ts, cpu_use=[u * 10 for u in BASE] + [950], vm_id=1)
    vm2 = make_vm(ts, cpu_use=[200.0] * len(ts), cores=[2] * 10 + [0] * 5,
                  cpu_cap=[1000.0] * 10 + [0.0] * 5, vm_id=2)
    return [vm1, vm2]


def test_run_writes_quality_logs_and_leaves_input_untouched(tmp_path):
    src = tmp_path / "loaded.csv"
    write_loaded_csv(src, two_vm_frames())
    before = cd.sha256_of(src)

    result = cd.run(src, tmp_path / "quality", expected_sha256=before)

    assert cd.sha256_of(src) == before
    for name, columns in ((cd.EVENTS_NAME, cd.EVENT_COLUMNS), (cd.SUMMARY_NAME, cd.SUMMARY_COLUMNS),
                          (cd.RUNS_NAME, cd.RUN_COLUMNS)):
        written = pd.read_csv(tmp_path / "quality" / name)
        assert list(written.columns) == columns
    assert not list((tmp_path / "quality").glob("*.tmp"))
    assert set(result["summary"]["vm_id"]) == {1, 2}
    assert result["summary"].groupby("vm_id").size().tolist() == [60, 60]
    assert events_of(result["events"], CPU_USE, "hampel_spike")["vm_id"].tolist() == [1]
    assert result["runs"][["vm_id", "resource", "rows"]].values.tolist() == [[2, "cpu", 5]]


def test_run_rejects_checksum_mismatch(tmp_path):
    src = tmp_path / "loaded.csv"
    write_loaded_csv(src, two_vm_frames())
    with pytest.raises(ValueError, match="체크섬 불일치"):
        cd.run(src, tmp_path / "quality", expected_sha256="0" * 64)
    assert not (tmp_path / "quality").exists()


def test_run_rejects_non_contiguous_vm_ids(tmp_path):
    src = tmp_path / "loaded.csv"
    vm1, vm2 = two_vm_frames()
    write_loaded_csv(src, [vm1.iloc[:5], vm2, vm1.iloc[5:]])
    with pytest.raises(ValueError, match="연속 구간"):
        cd.run(src, tmp_path / "quality", expected_sha256=None)
