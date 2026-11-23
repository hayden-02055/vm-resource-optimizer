# GWA-T-12 정제 슬라이스 A 동결 기록

> **동결 일자:** 2026-11-18
> **근거 전략:** `docs/GWA_T12_이상치_정제_전략.md` (v1.2)
> **입력 기준선:** `docs/GWA_T12_로딩_동결기록.md` — `data/vm_all_1250_loaded.csv`
> **의미:** 이 문서에 적힌 체크섬·행 수·스키마가 슬라이스 A(공통 시간 검사 + CPU/Memory) 판정 결과의 **확정 기준선**이다. 슬라이스 B 이후에도 Group A 부분이 바뀌지 않았는지 §4 절차로 대조한다.
> **적용 범위:** 판정 로그까지만. 정제 CSV(`data/vm_all_1250_clean.csv`)는 아직 생성하지 않았다(전략 §7.4). Group B, 결측 처리, 학습 데이터 구성은 포함하지 않는다.

## 1. 동결된 산출물

| 파일 | 바이트 | SHA-256 |
|---|---|---|
| `data/quality/anomaly_events.csv` | 139,569,529 | `57442af133e226a324631cdba55cb2bb77fd71d3fecc7f3187841d05b6c646be` |
| `data/quality/anomaly_summary.csv` | 2,281,491 | `3fca07acedfe64df2f6ef617d51cc275288e72210be7e3175e9ddb2561b59f63` |
| `data/quality/zero_allocation_runs.csv` | 20,850 | `af003b06d2fb7e1be108783d116be9a4955833ba72922227d1b15f8407bc0ff6` |
| `src/clean_data.py` | 15,505 | `e312ad4a469444fbd7b56bdc80b22d6f3adf4e4145059a0c0c5000e23483e17a` |
| `tests/test_clean_data.py` | 11,541 | `76e35c0685cbc79a02f7b4a50073311e75c5a396d4a79f04da926d3d590b4342` |
| `tests/conftest.py` | 109 | `de9e67117397d73b98592aa403b2110e07c1ff5ef2bb1aeca178d822d031832f` |
| `docs/GWA_T12_이상치_정제_전략.md` | 31,588 | `ab1515b521dd9f2a6190c2c8ff92d5270976d8c8465e097a2cf3f9d9242d3f17` |

입력 동결본과 로딩 단계 파일 4개(`data/vm_all_1250_loaded.csv`, manifest, `src/load_data.py`, 로딩 설계안)는 실행 전·후 모두 `docs/GWA_T12_로딩_동결기록.md` §1의 체크섬과 일치했다.

### Group A 부분 체크섬 (슬라이스 B 이후 대조용)

슬라이스 B는 같은 로그 파일에 `group = io` 행을 추가한다. 따라서 B 이후 **전체 체크섬이 바뀌는 것은 정상**이고, Group A 기준선은 **`io`가 아닌 행(헤더 포함)의 체크섬**으로 고정한다. 동결 시점에는 `io` 행이 없으므로 전체 체크섬과 같다.

| 대상 | 추출 규칙 | SHA-256 |
|---|---|---|
| `anomaly_events.csv` | 4번째 필드(`group`) ≠ `io` | `57442af133e226a324631cdba55cb2bb77fd71d3fecc7f3187841d05b6c646be` |
| `anomaly_summary.csv` | 3번째 필드(`group`) ≠ `io` | `3fca07acedfe64df2f6ef617d51cc275288e72210be7e3175e9ddb2561b59f63` |
| `zero_allocation_runs.csv` | 전체 (Group A 전용 파일) | `af003b06d2fb7e1be108783d116be9a4955833ba72922227d1b15f8407bc0ff6` |

두 로그 모두 콤마가 들어간 필드가 없어(모든 행의 필드 수가 각각 8개·6개로 일정) `awk -F,` 분리가 안전하다.

## 2. 동결된 데이터 형상

- **입력:** `data/vm_all_1250_loaded.csv` 11,221,800행 / VM 1,250개 / 중복 제거 시점 10,769,223개
- **인코딩:** UTF-8, BOM 없음(선두 `76 6d 5f`) / 줄바꿈 LF / 구분자 `,` / `.tmp` 잔여물 없음

| 파일 | 행 수 | 컬럼 |
|---|---|---|
| `anomaly_events.csv` | 1,699,455 | `vm_id`, `timestamp_raw`, `metric`, `group`, `status`, `reason`, `original_value`, `action` |
| `anomaly_summary.csv` | 40,000 (VM 1,250 × 32) | `vm_id`, `metric`, `group`, `status`, `reason`, `count` |
| `zero_allocation_runs.csv` | 446 | `vm_id`, `resource`, `start_timestamp_raw`, `end_timestamp_raw`, `rows`, `position`, `rows_usage_positive` |

### 이벤트 구성

| `metric` | `status` | `reason` | `action` | 행 수 |
|---|---|---|---|---|
| `CPU usage [MHZ]` | `SUSPECT` | `hampel_spike` | `keep` | 743,415 |
| `CPU usage [MHZ]` | `SUSPECT` | `over_capacity` | `keep` | 319,082 |
| `CPU usage [MHZ]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 40 (20시점) |
| `Memory usage [KB]` | `SUSPECT` | `hampel_spike` | `keep` | 635,835 |
| `Memory usage [KB]` | `SUSPECT` | `over_capacity` | `keep` | 628 |
| `Memory usage [KB]` | `UNDETERMINED` | `duplicate_conflict` | `keep` | 454 (239시점) |
| `CPU usage [%]` | `SUSPECT` | `pct_mismatch` | `keep` | 1 |

`INVALID`(`action = set_nan`) 이벤트는 **0행**이다. 이벤트가 하나 이상 있는 VM은 1,248개다.

### 집계 (전체 VM 합계)

판정 건수의 단위는 **시점**(중복 제거한 `(vm_id, timestamp)`)이다.

| `status` | `reason` | `CPU usage [MHZ]` | `Memory usage [KB]` |
|---|---|---|---|
| `VALID` | `hampel_normal` | 5,384,713 | 8,108,621 |
| `SUSPECT` | `hampel_spike` | 743,415 | 635,835 |
| `UNDETERMINED` | `insufficient_history` | 36,981 | 36,979 |
| `UNDETERMINED` | `zero_mad` | 3,463,100 | 849,436 |
| `UNDETERMINED` | `zero_allocation` | 1,140,994 | 1,138,113 |
| `UNDETERMINED` | `duplicate_conflict` | 20 | 239 |
| | **분할 합계** | **10,769,223** | **10,769,223** |
| `SUSPECT` | `over_capacity` (분할과 별개) | 319,082 | 628 |

- `CPU usage [%]`: `pct_match` 9,628,228 / `pct_mismatch` 1
- `INVALID` (`negative_value`, `non_finite`): Group A 6개 컬럼 모두 **0**
- `Timestamp [ms]`(`common`, 단위: 인접 간격): `regular_step` 9,463,104 / `duplicate_timestamp` 452,577 / `irregular_step` 1,299,911 / `segment_break` 4,958 → 합계 11,220,550

## 3. 검수 결과 (11/11 통과)

| # | 항목 | 결과 |
|---|---|---|
| 1 | 단위 테스트 `tests/test_clean_data.py` | 18 / 18 통과 |
| 2 | 회귀 테스트 유효성 | MAD를 `np.median`으로 되돌린 변형 코드에 같은 입력을 넣으면 급증을 놓친다(판정 코드 `NOT_EVALUATED`) → 회귀 테스트가 이 버그를 잡는다 |
| 3 | lint | `ruff check src/clean_data.py tests/` 통과 |
| 4 | 입력 동결본 체크섬 | 실행 전·후 모두 `ee2426537153…` 일치 |
| 5 | 재현성 | 같은 입력·코드로 다시 실행한 3개 파일이 바이트 단위로 동일 |
| 6 | 시간 간격 분할 | 4개 분류 합 11,220,550 = 행 수 − VM 수 |
| 7 | 판정 분할 (CPU, Memory 각각) | `over_capacity`를 뺀 분류 합 10,769,223 = 중복 제거 시점 수 |
| 8 | 집계 커버리지 | VM 1,250개 × 32행, 0건인 검사 포함 |
| 9 | 할당 0 구간 행 수 | CPU 구간 행 합 1,593,544 = `CPU cores = 0` 행 수 |
| 10 | 값 변경 대상 | `INVALID` 0건 → Group A가 바꾸는 값 없음 |
| 11 | 파일 위생 | BOM 없음 / `.tmp` 잔여물 없음 / 필드 수 일정 |

**실행 환경:** Python 3.13.2, pandas 2.3.3, numpy 2.3.1, pytest 8.3.2, ruff 0.17.5. 실행 시간 약 29초. 다른 버전에서는 부동소수 표기가 달라져 바이트가 다를 수 있으므로, 그때는 값 동일성을 따로 확인한다.

## 4. 재현 및 무결성 대조 절차

```bash
# (a) 현재 파일이 동결 시점과 같은지 확인 (슬라이스 B 이전)
shasum -a 256 data/quality/anomaly_events.csv data/quality/anomaly_summary.csv \
              data/quality/zero_allocation_runs.csv src/clean_data.py \
              tests/test_clean_data.py tests/conftest.py docs/GWA_T12_이상치_정제_전략.md

# (b) Group A 부분 대조 — 슬라이스 B 이후에도 §1 부분 체크섬과 같아야 한다
awk -F, '$4!="io"' data/quality/anomaly_events.csv  | shasum -a 256
awk -F, '$3!="io"' data/quality/anomaly_summary.csv | shasum -a 256
shasum -a 256 data/quality/zero_allocation_runs.csv

# (c) 동결본에서 다시 생성 (출력은 동일 체크섬이어야 한다)
python3 src/clean_data.py
python3 -m pytest tests -q
```

`src/clean_data.py`는 입력 동결본을 읽기만 하고, 같은 입력과 같은 스크립트는 같은 바이트 출력을 낸다(§3-5). (a)가 달라졌다면 로그나 스크립트가 바뀐 것이고, B 이후 (b)가 달라졌다면 **Group A 판정 결과가 바뀐 것**이다.

## 5. 동결 시점에 관측된 사실 (해석·처리는 미정)

판정 결과에서 확인된 값들이다. **어떤 값도 변경하지 않았다.**

1. **Hampel 급증 비율이 높다.** `hampel_spike`는 판정이 성립한 시점 대비 CPU **12.13%**, Memory **7.27%**다(전체 시점 대비 6.90% / 5.90%). 임계값 3σ 상향 검토 대상이다.
2. **판정이 한 번도 성립하지 않은 VM**은 CPU 53개, Memory 13개다. 주된 사유는 `zero_mad`(CPU 42·Memory 7), `zero_allocation`(8·4), `insufficient_history`(3·2) 순이다.
3. **할당 0 구간은 모두 VM 선두·후미에 있다.** 중간 구간은 0개다.

   | 자원 | 위치 | 구간 수 | 행 수 | 할당 0인데 사용량 > 0 |
   |---|---|---|---|---|
   | cpu | leading | 103 | 228,106 | 0 |
   | cpu | trailing | 121 | 1,365,438 | 11,993 |
   | memory | leading | 102 | 228,113 | 1 |
   | memory | trailing | 120 | 1,361,095 | 46,963 |

   해당 VM은 CPU 210개, Memory 208개다.
4. **중복 시각:** 인접 간격 452,577개가 0이다(VM 122개). 사용률 값이 충돌한 시점은 CPU 20개, Memory 239개다.
5. **시간 간격:** 300초가 9,463,104개(84.3%)다. 1~298초 간격 909,264개는 중복 VM 122개에서만 나오고, 299·301초(±1초)는 거의 모든 VM에 있다. 600초 초과는 4,958개이며 최대 간격은 2,100초다.
6. **`CPU usage [%]`는 사실상 중복이다.** 할당 > 0인 9,628,229시점 중 0.01%p 넘게 어긋난 것은 1개뿐이다(vm 752, `timestamp_raw` 1378487659, 기록값 48.677083%로 계산값 약 97.35%의 절반). 제거 여부는 미정이다.
7. **할당량 초과:** CPU 319,082시점(최대 186.6%), Memory 628시점(최대 307,200%, 예: 할당 4,096KB에 사용 12,582,912KB). 값은 유지했다.

## 6. 동결 이후 변경 규칙

- `data/vm_all_1250_loaded.csv`는 계속 **읽기 전용**이다(로딩 동결기록 §6).
- 슬라이스 B는 `data/quality/` 3개 파일을 **같은 경로에 다시 만든다**(전략 §7.4). 이때 전체 체크섬은 바뀌어도 되지만 **§1의 Group A 부분 체크섬은 바뀌면 안 된다.**
- 이를 위해 슬라이스 B는 Group A 판정 로직과 출력 형식을 바꾸지 않는다. VM마다 Group A 행 **뒤에** `group = io` 행을 덧붙여 Group A 행의 상대 순서를 유지한다.
- Group A 규칙(임계값 등)을 바꿀 때는 이 문서를 지우지 않고 **새 동결 항목을 추가**하며, 변경 전후 집계 차이를 함께 적는다.
- 정제 CSV(`data/vm_all_1250_clean.csv`)를 생성하면 별도 동결 기록을 추가한다.
