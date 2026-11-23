"""src/build_clean_csv.py 테스트 — set_nan 적용 범위와 무결성 검사를 합성 데이터로 검증한다."""
import numpy as np
import pandas as pd
import pytest

import build_clean_csv as bc
from clean_data import CORES, CPU_USE, EVENT_COLUMNS, READ_COLUMNS, TS, sha256_of


def loaded_frame():
    df = pd.DataFrame({col: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for col in READ_COLUMNS})
    df["vm_id"] = [1, 1, 1, 2, 2, 2]
    df[TS] = pd.array([0, 300, 300, 0, 300, 600], dtype="Int64")      # vm 1의 시각 300이 중복
    df[CORES] = pd.array([2, 2, 2, 4, -1, 4], dtype="Int64")
    df[CPU_USE] = [100.0, 200.0, -5.0, 300.0, 6.7108864e7, 0.1]
    return df


def event(vm_id, ts, metric, value, action="set_nan"):
    keep = action == "keep"
    return {"vm_id": vm_id, "timestamp_raw": ts, "metric": metric, "group": "resource",
            "status": "SUSPECT" if keep else "INVALID",
            "reason": "hampel_spike" if keep else "negative_value",
            "original_value": value, "action": action}


def write_inputs(tmp_path, events):
    src, ev = tmp_path / "loaded.csv", tmp_path / "events.csv"
    loaded_frame().to_csv(src, index=False)
    pd.DataFrame(events, columns=EVENT_COLUMNS).to_csv(ev, index=False)
    return src, ev


def build(tmp_path, events):
    src, ev = write_inputs(tmp_path, events)
    out = tmp_path / "clean.csv"
    result = bc.build_clean_csv(src, ev, out, expected_sha256=None, expected_events_sha256=None)
    return src, out, result


def test_without_set_nan_events_output_is_byte_identical(tmp_path):
    src, out, result = build(tmp_path, [event(1, 300, CPU_USE, 200.0, action="keep")])
    assert result["changed"] == 0
    assert result["byte_identical"] and sha256_of(out) == sha256_of(src)


def test_set_nan_changes_only_the_matching_duplicate_cell(tmp_path):
    _, out, result = build(tmp_path, [event(1, 300, CPU_USE, -5.0)])

    expected = loaded_frame()
    expected.loc[2, CPU_USE] = np.nan                              # 같은 시각의 200.0(행 1)은 유지
    pd.testing.assert_frame_equal(pd.read_csv(out, **bc.READ_OPTIONS), expected)
    assert result["changed"] == 1 and not result["byte_identical"]
    assert result["comparison"].loc["missing cells"].tolist() == [0, 1]


def test_set_nan_on_int64_column_keeps_dtype(tmp_path):
    _, out, result = build(tmp_path, [event(2, 300, CORES, -1.0)])

    written = pd.read_csv(out, **bc.READ_OPTIONS)
    assert written[CORES].dtype == "Int64" and pd.isna(written.loc[4, CORES])
    assert out.read_text().splitlines()[5].split(",")[2] == ""     # CSV에는 빈 필드로 기록
    assert result["changed"] == 1


def test_event_without_matching_cell_raises(tmp_path):
    with pytest.raises(ValueError, match="찾을 수 없는"):
        build(tmp_path, [event(1, 300, CPU_USE, 999.0)])
    assert not (tmp_path / "clean.csv").exists()


def test_unknown_metric_raises(tmp_path):
    with pytest.raises(ValueError, match="데이터 컬럼이 아닙니다"):
        build(tmp_path, [event(1, 0, "Bogus [x]", 1.0)])


def test_refuses_to_overwrite_input(tmp_path):
    src, ev = write_inputs(tmp_path, [])
    with pytest.raises(ValueError, match="덮어쓸 수 없습니다"):
        bc.build_clean_csv(src, ev, src, expected_sha256=None, expected_events_sha256=None)


@pytest.mark.parametrize("which", ["input", "events"])
def test_checksum_mismatch_raises_before_writing(tmp_path, which):
    src, ev = write_inputs(tmp_path, [])
    out = tmp_path / "clean.csv"
    kwargs = {"expected_sha256": None, "expected_events_sha256": None}
    kwargs["expected_sha256" if which == "input" else "expected_events_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="체크섬 불일치"):
        bc.build_clean_csv(src, ev, out, **kwargs)
    assert not out.exists()
