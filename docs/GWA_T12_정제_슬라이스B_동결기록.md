# GWA-T-12 정제 슬라이스 B 동결 기록

> **동결 일자:** 2026-11-19 (산출물 생성 2026-11-18)
> **근거 전략:** `docs/GWA_T12_이상치_정제_전략.md` (v1.3)
> **선행 기준선:** `docs/GWA_T12_로딩_동결기록.md`(입력), `docs/GWA_T12_정제_슬라이스A_동결기록.md`(Group A)
> **의미:** 이 문서에 적힌 체크섬·행 수·스키마가 슬라이스 A·B(공통 시간 검사 + CPU/Memory + Disk/Network) 판정 결과의 **확정 기준선**이다. 슬라이스 A의 Group A 기준선은 이 문서에서도 그대로 유효하다(§3-5).
> **적용 범위:** 판정 로그까지만. 정제 CSV, 결측 처리, 학습 데이터 구성은 포함하지 않는다.

## 1. 동결된 산출물

| 파일 | 바이트 | SHA-256 |
|---|---|---|
| `data/quality/anomaly_events.csv` | 240,653,978 | `a6abfeff5bf50609913d03142fc856e5eb497abcbd310c11826f878bab662589` |
| `data/quality/anomaly_summary.csv` | 4,624,699 | `45dd50badd60adba25be84beb0ef13a207700be19f4a0d5b101fcbc28a388b5b` |
| `data/quality/zero_allocation_runs.csv` | 20,850 | `af003b06d2fb7e1be108783d116be9a4955833ba72922227d1b15f8407bc0ff6` |
| `src/clean_data.py` | 17,901 | `af37ba2d8efe0da0ed388e5d800c9b8cf564d6daa8033b27bd22a185676da922` |
| `tests/test_clean_data.py` | 15,589 | `8b41180c09d03d71781ae071a3f51c2541e2d1d0a01c94bbd15ee8e3da1f85c2` |
| `tests/conftest.py` | 109 | `de9e67117397d73b98592aa403b2110e07c1ff5ef2bb1aeca178d822d031832f` |
| `docs/GWA_T12_이상치_정제_전략.md` | 33,820 | `14fb0b22ebc1531722bc0075a4fa6b6c00226a26a7be18daab03b3ee16d29c30` |

슬라이스 A 동결기록 §1과 비교하면 `zero_allocation_runs.csv`와 `tests/conftest.py`는 바이트까지 같고, 나머지 5개는 B를 추가하면서 바뀌었다(슬라이스 A 동결기록 §6에서 예정된 변경). 입력 동결본(`data/vm_all_1250_loaded.csv`)은 실행 전·후 모두 로딩 동결기록 §1의 체크섬과 일치했다.

### 그룹별 부분 체크섬

판정 규칙을 나중에 바꿀 때, **바꾼 그룹만 달라지고 다른 그룹은 그대로인지** 확인하기 위한 기준선이다.

| 대상 | 추출 규칙 | SHA-256 | 비고 |
|---|---|---|---|
| events — Group A | 4번째 필드(`group`) ≠ `io` | `57442af133e226a324631cdba55cb2bb77fd71d3fecc7f3187841d05b6c646be` | 슬라이스 A 동결값과 **일치** |
| summary — Group A | 3번째 필드(`group`) ≠ `io` | `3fca07acedfe64df2f6ef617d51cc275288e72210be7e3175e9ddb2561b59f63` | 슬라이스 A 동결값과 **일치** |
| events — Group B | 헤더 + 4번째 필드 = `io` | `cfe70b771280b7cc283b3d1fca76510fbdcc00f4f996a6079643baf597393e78` | 1,119,963행 + 헤더 |
| summary — Group B | 헤더 + 3번째 필드 = `io` | `372a08a1a05a8ad3b88841ff3f0a31a86dea212b31fef57ff117cd6eb67fbaa5` | 35,000행 + 헤더 |
| `zero_allocation_runs.csv` | 전체 (Group A 전용) | `af003b06d2fb7e1be108783d116be9a4955833ba72922227d1b15f8407bc0ff6` | 슬라이스 A 동결값과 **일치** |

두 로그 모두 콤마가 들어간 필드가 없어(모든 행의 필드 수가 각각 8개·6개로 일정) `awk -F,` 분리가 안전하다.

## 2. 동결된 데이터 형상

- **입력:** `data/vm_all_1250_loaded.csv` 11,221,800행 / VM 1,250개 / 중복 제거 시점 10,769,223개
- **인코딩:** UTF-8, BOM 없음(선두 `76 6d 5f`) / 줄바꿈 LF(CR 0개) / 구분자 `,` / `.tmp` 잔여물 없음

| 파일 | 행 수 | 그중 Group B | 컬럼 |
|---|---|---|---|
| `anomaly_events.csv` | 2,819,418 | 1,119,963 | `vm_id`, `timestamp_raw`, `metric`, `group`, `status`, `reason`, `original_value`, `action` |
| `anomaly_summary.csv` | 75,000 (VM 1,250 × 60) | 35,000 (VM 1,250 × 28) | `vm_id`, `metric`, `group`, `status`, `reason`, `count` |
| `zero_allocation_runs.csv` | 446 | — | `vm_id`, `resource`, `start_timestamp_raw`, `end_timestamp_raw`, `rows`, `position`, `rows_usage_positive` |

### Group B 이벤트 구성

| `metric` | `status` | `reason` | `action` | 행 수 |
|---|---|---|---|---|
| `Disk read throughput [KB/s]` | `SUSPECT` | `hampel_spike` | `keep` | 66,325 |
| `Disk write throughput [KB/s]` | `SUSPECT` | `hampel_spike` | `keep` | 586,513 |
| `Network received throughput [KB/s]` | `SUSPECT` | `hampel_spike` | `keep` | 235,032 |
| `Network transmitted throughput [KB/s]` | `SUSPECT` | `hampel_spike` | `keep` | 231,399 |
| `Disk read throughput [KB/s]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 22 (11시점) |
| `Disk write throughput [KB/s]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 60 (30시점) |
| `Network received throughput [KB/s]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 282 (141시점) |
| `Network transmitted throughput [KB/s]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 330 (165시점) |

Group B의 `INVALID`(`action = set_nan`) 이벤트는 **0행**이다. 이벤트 전체로 보면 `SUSPECT` 2,818,230행(A 1,698,961 + B 1,119,269), `UNDETERMINED` 1,188행(A 494 + B 694), `INVALID` 0행이다. Group B 이벤트가 있는 VM은 1,192개, 이벤트가 하나라도 있는 VM은 1,249개다.

### Group B 집계 (전체 VM 합계)

판정 건수의 단위는 **시점**(중복 제거한 `(vm_id, timestamp)`)이다.

| `status` | `reason` | Disk read | Disk write | Network received | Network transmitted |
|---|---|---|---|---|---|
| `VALID` | `hampel_normal` | 705,976 | 7,430,603 | 4,313,992 | 3,659,380 |
| `SUSPECT` | `hampel_spike` | 66,325 | 586,513 | 235,032 | 231,399 |
| `UNDETERMINED` | `insufficient_history` | 62,512 | 62,524 | 62,512 | 62,512 |
| `UNDETERMINED` | `zero_mad` | 9,934,399 | 2,689,553 | 6,157,546 | 6,815,767 |
| `UNDETERMINED` | `duplicate_conflict` | 11 | 30 | 141 | 165 |
| | **분할 합계** | **10,769,223** | **10,769,223** | **10,769,223** | **10,769,223** |
| `INVALID` | `negative_value` / `non_finite` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

Group A 집계는 슬라이스 A 동결기록 §2와 같다(부분 체크섬 일치로 확인, §3-5).

## 3. 검수 결과 (11/11 통과)

| # | 항목 | 결과 |
|---|---|---|
| 1 | 단위 테스트 `tests/test_clean_data.py` | 24 / 24 통과 (A 18 + B 6: 상방 단측, 원래 처리량 기록, 0 이후 첫 양수 미판정, 음수·inf `INVALID`, 지표별 값 충돌, A 출력의 I/O 독립성·순서) |
| 2 | lint | `ruff check src/clean_data.py tests/` 통과 |
| 3 | 입력 동결본 체크섬 | 실행 전·후 모두 `ee2426537153…` 일치 |
| 4 | 재현성 | 같은 입력·코드로 다시 실행한 3개 파일이 바이트 단위로 동일 |
| 5 | Group A 불변 | Group A 부분 체크섬 3개가 슬라이스 A 동결값과 일치 |
| 6 | 순서 제약 | 이벤트·집계 파일 모두 VM마다 `io` 행이 그 VM의 Group A 행 뒤에만 있고, `vm_id`는 비감소 (전수 확인) |
| 7 | Group B 판정 분할 | 4개 지표 각각 분류 합 10,769,223 = 중복 제거 시점 수 |
| 8 | 집계 커버리지 | VM 1,250개 × 60행(그중 B 28행), 0건인 검사 포함 |
| 9 | 이벤트 수 대조 | B `hampel_spike` 이벤트 1,119,269행 = 집계 합계 / 충돌 이벤트 694행 = 충돌 시점 347개 × 값 2개 |
| 10 | 값 변경 대상 | `INVALID` A·B 모두 0건 → `set_nan` 이벤트 0행 |
| 11 | 파일 위생 | BOM 없음 / CR 없음 / `.tmp` 잔여물 없음 / 필드 수 일정 |

**실행 환경:** Python 3.13.2, pandas 2.3.3, numpy 2.3.1, pytest 8.3.2, ruff 0.17.5. 실행 시간 약 71초. 다른 버전에서는 부동소수 표기가 달라져 바이트가 다를 수 있으므로, 그때는 값 동일성을 따로 확인한다.

## 4. 재현 및 무결성 대조 절차

```bash
# (a) 현재 파일이 동결 시점과 같은지 확인
shasum -a 256 data/quality/anomaly_events.csv data/quality/anomaly_summary.csv \
              data/quality/zero_allocation_runs.csv src/clean_data.py \
              tests/test_clean_data.py tests/conftest.py docs/GWA_T12_이상치_정제_전략.md

# (b) Group A 부분 — 슬라이스 A 동결값과도 같아야 한다
awk -F, '$4!="io"' data/quality/anomaly_events.csv  | shasum -a 256
awk -F, '$3!="io"' data/quality/anomaly_summary.csv | shasum -a 256
shasum -a 256 data/quality/zero_allocation_runs.csv

# (c) Group B 부분
awk -F, 'NR==1 || $4=="io"' data/quality/anomaly_events.csv  | shasum -a 256
awk -F, 'NR==1 || $3=="io"' data/quality/anomaly_summary.csv | shasum -a 256

# (d) 동결본에서 다시 생성 (출력은 동일 체크섬이어야 한다)
python3 src/clean_data.py
python3 -m pytest tests -q
```

(a)가 달라졌다면 로그·코드·전략 문서 중 무언가 바뀐 것이다. 이때 (b)가 달라졌다면 **Group A 판정 결과**가, (c)가 달라졌다면 **Group B 판정 결과**가 바뀐 것이다.

## 5. 동결 시점에 관측된 사실 (해석·처리는 미정)

판정 결과와 동결본에서 확인된 값들이다. **어떤 값도 변경하지 않았다.**

1. **값 변경 대상이 없다.** `INVALID`가 Group A·B 모두 0건이다. 따라서 전략 §7.4대로 정제 CSV를 만들면 값은 동결본과 같다.
2. **Group B 급증 비율:** `hampel_spike`는 판정이 성립한 시점 대비 Disk read 8.59%, Disk write 7.32%, Network received 5.17%, Network transmitted 5.95%다(전체 시점 대비 0.62% / 5.45% / 2.18% / 2.15%). 임계값 6σ 상향 검토 대상이다.
3. **`zero_mad`가 가장 많다.** 전체 시점 대비 Disk read 92.25%, Disk write 24.97%, Network received 57.18%, Network transmitted 63.29%다. 판정이 한 번도 성립하지 않은 VM은 각각 736·51·259·238개이며, **Disk read는 대부분의 VM에서 통계 탐지가 사실상 작동하지 않는다.**
4. **0 비율(동결본 행 기준):** Disk read 89.4%, Disk write 22.6%, Network received 54.1%, Network transmitted 51.2%다. 모든 행이 0인 VM이 각각 254·15·65·110개다. 0은 전략 §5.2대로 유효값으로 유지했다.
5. **할당 0 시점은 B 판정에 포함했다**(전략 v1.3). 착수 전 확인에서 이 시점의 처리량은 96~99.7%가 0이었고, 제외했을 때와의 급증 후보 수 차이는 지표당 14개 이하였다.
6. **중복 시각 값 충돌:** Disk read 11, Disk write 30, Network received 141, Network transmitted 165시점이며, 모두 서로 다른 값 2개가 기록되어 있다.

## 6. 동결 이후 변경 규칙

- `data/vm_all_1250_loaded.csv`는 계속 **읽기 전용**이다(로딩 동결기록 §6).
- `data/quality/` 3개 파일의 현재 기준선은 **이 문서 §1**이다. 슬라이스 A 동결기록 §4(a)는 B 이전 기준이므로 현재 파일과 맞지 않는 것이 정상이다. 그 문서의 Group A 부분 체크섬(§4(b))은 계속 유효하다.
- 정제 CSV를 만드는 단계는 품질 로그를 **읽기만** 하고 다시 쓰지 않는다. 정제 CSV는 별도 동결 기록으로 남긴다.
- 판정 규칙(임계값 등)을 바꿀 때는 이 문서와 슬라이스 A 동결기록을 지우지 않고 **새 동결 항목을 추가**한다. 바꾼 그룹의 부분 체크섬만 달라지고 다른 그룹의 부분 체크섬은 그대로여야 하며, 변경 전후 집계 차이를 함께 적는다.
