# GWA-T-12 이상치 정제 전략 v1.3

> **프로젝트:** 머신러닝 기반 가상머신 자원 사용량 예측 및 동적 자원 할당 시스템  
> **대상:** GWA-T-12 `fastStorage` 로딩 동결본 `data/vm_all_1250_loaded.csv` (11,221,800행 × 12컬럼)  
> **기준선:** `docs/GWA_T12_로딩_동결기록.md` — 입력 파일의 체크섬·스키마·dtype은 이 문서를 따른다.  
> **도구:** `pandas`, `numpy`  
> **이번 문서의 범위:** 이상치 탐지·판정·처리. 결측값 보간, 정규화, 모델 학습, 동적 할당 알고리즘은 별도 단계에서 결정한다.  
> **상태:** v1.2에서 **Group A**, v1.3에서 **Group B** 처리 정책 확정(2026-11-18, 동결본 전수 확인 반영). 통계 임계값(window·threshold)은 여전히 **초기값**이며 확정된 최적값이 아니다.

## 1. 결론 — 정제 원칙

**목표는 극단값을 없애는 것이 아니라, 측정 오류를 줄이면서 실제 자원 사용량의 급증·급락을 보존하는 것이다.** 동적 자원 할당 시스템에서 급격한 부하 변화는 예측해야 할 핵심 신호일 수 있다.

1. **탐지와 처리를 분리**한다. 통계적으로 희귀하다는 이유만으로 관측치를 삭제하거나 평균값으로 바꾸지 않는다.
2. **VM별·시간순**으로 판단한다. 서로 다른 VM의 사용량을 한꺼번에 비교하지 않는다.
3. **CPU·Memory(할당 용량 대비 사용 상태)**와 **Disk·Network(입출력 처리량)**에 서로 다른 통계적 규칙을 적용한다.
4. **확실한 비정상 수치만 해당 셀을 `NaN`으로 표시**한다. 행 전체 삭제, 상·하위 값 절단, 임의 클리핑은 기본적으로 하지 않는다.
5. **`Timestamp [ms]` 원본 숫자는 그대로 보존**한다. 시간 순서·간격 검사는 하되 값 자체를 재작성하지 않는다. 동결기록 §5-3에 따르면 값은 **초 단위 Unix 시간**으로 해석된다(범위 `1376314846 ~ 1378906798` = 2013-08-12 ~ 2013-09-11 UTC, 30일). 따라서 샘플의 인접 간격 `300`은 **300초(5분)**다. 전체 데이터의 실제 기록 간격은 동결 시점에 검증되지 않았으므로 **정제 첫 단계에서 VM별로 확인**한다. 단위 표기 확정과 컬럼명 `[ms]` 정정 여부는 이 문서의 범위 밖이다(미결).
6. **`CPU cores`와 capacity 값은 유지**한다. vCPU 수와 할당 용량이 VM별로 다르며, 동일 VM에서도 변할 가능성이 있으므로 변화 자체를 오류로 판단하지 않는다. 값이 `0`인 할당도 오류가 아닌 상태로 보고 유지한다(§3.1).
7. **원본·정제 결과·판정 로그를 구분**한다. 모델 입력 테이블에 수십 개의 플래그 컬럼을 억지로 추가하지 않고 별도 품질 보고서에 기록한다.

### 처리 결과 분류

| 분류 | 의미 | 자동 처리 |
|---|---|---|
| `VALID` | 범위와 시계열 특성상 오류 근거 없음 | 값 유지 |
| `INVALID` | 음수 처리량, 유효하지 않은 용량 등 명백한 형식·물리적 오류 | **해당 값만 `NaN`**으로 표시하고 로그 기록 |
| `SUSPECT` | 주변 관측값에 비해 이례적인 변화, 할당 용량 초과 등 추가 확인 필요 | **원본 값 유지**, 로그 기록 |
| `UNDETERMINED` | 데이터 부족, MAD=0, 할당 0, 값이 다른 중복 시각, 불규칙한 시간 간격 등으로 판정 곤란 | 값 유지, 사유 기록 |

> **중요:** `SUSPECT`는 삭제 대상이 아니다. `INVALID`로 판정한 수치도 원본 파일에서 지우지 않고 정제본의 해당 값만 변경한다. 이 과정에서 발생한 결측값을 어떻게 채울지는 다음 단계에서 결정한다.

---

## 2. 데이터 그룹과 적용 대상

로딩 설계안에서 정한 두 그룹을 **논리적으로만 구분**한다. CSV 파일을 물리적으로 두 개로 나누지는 않는다.

| 그룹 | 컬럼 | 이상치 처리 시의 역할 |
|---|---|---|
| **A — 자원 사용 상태** | `CPU cores` | vCPU 사양·할당 변화 검사. Hampel 대상 **아님** |
| | `CPU capacity provisioned [MHZ]` | 할당 CPU 용량 유효성 검사. Hampel 대상 **아님** |
| | `CPU usage [MHZ]` | 실제 CPU 사용량 검사. **사용률로 환산해 Hampel 탐지** |
| | `CPU usage [%]` | 다른 두 CPU 컬럼과 일치 여부 검증 후 중복이면 제거 후보 |
| | `Memory capacity provisioned [KB]` | 할당 메모리 용량 유효성 검사. Hampel 대상 **아님** |
| | `Memory usage [KB]` | 실제 메모리 사용량 검사. **사용률로 환산해 Hampel 탐지** |
| **B — I/O 처리량** | `Disk read throughput [KB/s]` | 비음수 검증 + **보수적 급증 탐지** |
| | `Disk write throughput [KB/s]` | 비음수 검증 + **보수적 급증 탐지** |
| | `Network received throughput [KB/s]` | 비음수 검증 + **보수적 급증 탐지** |
| | `Network transmitted throughput [KB/s]` | 비음수 검증 + **보수적 급증 탐지** |
| **공통** | `Timestamp [ms]`, `vm_id` | 시계열 순서·연속성 및 VM 식별. 수치 이상치 필터 대상 **아님** |

### CPU·Memory를 비율로 비교하는 이유

VM마다 할당 용량이 다르므로 **절대 사용량이 큰 VM = 이상한 VM**은 아니다. 시계열 이상치 후보를 탐지할 때는 다음 **임시 계산값**을 사용한다.

- `cpu_util_pct = CPU usage [MHZ] / CPU capacity provisioned [MHZ] × 100`
- `memory_util_pct = Memory usage [KB] / Memory capacity provisioned [KB] × 100`

두 비율은 판정에 활용하지만 **정제본에 반드시 새 컬럼으로 저장할 필요는 없다**. 예측 타깃인 절대 사용량(`MHz`, `KB`)은 원래 컬럼 그대로 유지한다.

---

## 3. 공통 1차 검사 — 통계보다 먼저

데이터 전체에 적용하는 규칙이다. 단, **확실한 오류와 측정 정의가 애매한 사례를 구분**한다.

| 검사 항목 | 조건 | 판정 및 처리 |
|---|---|---|
| 숫자 형식 | 숫자로 변환 불가, `+inf`/`-inf` | `INVALID` → 해당 셀 `NaN` |
| vCPU 수 | `CPU cores`가 음수 또는 정수가 아님 | `INVALID` → 해당 셀 `NaN` |
| CPU 할당 용량 | `CPU capacity provisioned [MHZ] < 0` | `INVALID` → 해당 셀 `NaN` |
| 메모리 할당 용량 | `Memory capacity provisioned [KB] < 0` | `INVALID` → 해당 셀 `NaN` |
| 할당 0 | `CPU cores`·할당 용량이 `0` | **오류 아님 → 값 유지**. 할당 0 구간으로 기록(§3.1) |
| CPU·메모리 사용량 | 사용량이 음수 | `INVALID` → 해당 셀 `NaN` |
| Disk·Network 처리량 | 처리량이 음수 | `INVALID` → 해당 셀 `NaN` |
| 사용량 / 할당 용량 | 할당 용량 > 0인데 사용량이 할당량보다 큼 | **즉시 삭제 금지** → `SUSPECT`(`over_capacity`); 메트릭 정의·수집 시점 차이 확인 |
| 높은 CPU·Memory 사용률 | 95~100% 등 | **정상 가능** → 값 유지 |
| Disk·Network 처리량 `0` | 전송·읽기·쓰기 활동 없음 | **정상 가능** → 값 유지 |
| 동일 VM의 timestamp 중복 | 동일 `(vm_id, timestamp)` 2행 이상 | **행 유지**. 판정은 중복 제거 시계열로 하고 결과는 같은 시각의 모든 행에 적용(§3.2). **다른 VM의 동일 시각은 중복 아님** |
| 불규칙 시간 간격 | VM 내 간격이 300초가 아님 | 건수 기록, 보간·행 삭제는 보류. **간격 > 600초면 새 구간**으로 보고 Hampel 이력을 다시 쌓음(§4.2) |

**할당량 초과 유의:** CPU/메모리 사용량과 provisioned capacity의 측정 정의가 다르거나 수집 시점이 어긋날 수 있다. `usage > capacity`만으로 물리적으로 불가능하다고 단정하지 않는다. 할당 용량이 0인 행은 비율을 정의할 수 없으므로 이 검사에서 빼고, 사용량이 남아 있는 행 수만 §3.1 구간 기록에 남긴다. 동결본 사전 확인: CPU 319,082행(최대 186.6%), Memory 628행(최대 307,200%, 예: 할당 4,096KB에 사용 12,582,912KB).

**스키마 중복 유의:** `CPU usage [%]`가 `CPU usage [MHZ] / CPU capacity provisioned [MHZ] × 100`과 일치하는지 **전체 원본에서 허용 오차를 두고 검사**한다. 일치하면 이 컬럼은 모델 입력에서 제거할 수 있지만, 이는 **이상치 처리와 별개의 스키마 정리**다. **v1.2 검사 규칙:** CPU 할당 용량 > 0인 행에서 `|CPU usage [%] − 계산값| > 0.01`%p이면 `SUSPECT`(`pct_mismatch`)로 기록한다. 동결본 사전 확인 결과 해당 행은 1개(1e-6 초과는 181행)여서 사실상 중복이다. 제거 결정은 여전히 별도다.

### 3.1 할당 0 구간 (v1.2 확정)

- **사실(동결본 전수):** `CPU cores = 0`과 `CPU capacity = 0`은 항상 같은 행에서 나타난다(1,593,544행, VM 210개). 그중 1,589,200행은 `Memory capacity = 0`도 동시에 해당한다. CPU 할당 0 구간은 224개이고 **모두 VM의 처음 또는 끝에 붙어 있다**: VM 첫 행부터 이어지는 구간 103개(228,106행), VM 끝까지 이어지는 구간 121개(1,365,438행). 중간에 끼어 있는 구간은 없다(14개 VM은 앞뒤 모두 0).
- **판단:** 측정 오류가 아니라 VM이 트레이스 기간 중 **아직 생성되지 않았거나 이미 종료된 할당 없음 상태**로 본다. → 값 유지(`NaN` 변환 안 함).
- **처리:** 사용률을 정의할 수 없으므로 Hampel 판정과 할당량 초과 판정에서 빼고, 집계에는 `UNDETERMINED`/`zero_allocation`으로 센다. 연속 구간은 자원별(`cpu`: `CPU cores = 0` 또는 CPU 할당 0, `memory`: Memory 할당 0)로 `data/quality/zero_allocation_runs.csv`에 기록한다.
- **미결:** 이 구간을 학습에서 제외할지는 다음 단계(학습 데이터 구성)에서 정한다.

### 3.2 중복 timestamp (v1.2 확정)

- **사실(동결본 전수):** VM 122개, 450,211그룹(902,788행. 2행 그룹 447,845개, 3행 그룹 2,366개). 449,968그룹(99.95%)은 Group A 값까지 같다. 중복 행은 모두 인접해 있고 VM 안에서 시간이 역행하는 경우는 없다.
- **처리:** 행은 지우지 않는다. 판정용 시계열은 `(vm_id, timestamp)`별 첫 행으로 만들고, 이벤트는 `(vm_id, timestamp_raw, metric)` 키로 남기므로 결과가 같은 시각의 모든 행에 적용된다.
- **값 충돌:** 같은 시각 안에서 해당 지표의 사용률이 서로 다르면 `UNDETERMINED`(`duplicate_conflict`)로 기록하고, 그 시점은 Hampel 이력에서 뺀다.
- **참고:** 중복이 있는 VM은 중복을 제거해도 300초 간격 비율이 37%에 그친다(`7`초 + `293`초처럼 어긋난 간격). 1~298초 간격 909,264개는 **이 122개 VM에서만** 나타나며, 299·301초(±1초 흔들림)는 거의 모든 VM에 있다. 이번 단계에서는 건수만 기록하고 보정하지 않는다.

---

## 4. 그룹 A — CPU / Memory 이상치 처리 정책

### 4.1 무엇을 잡을 것인가?

- 정상적인 고부하(예: CPU 98%) 자체를 이상치로 보지 않는다.
- **해당 VM의 최근 사용 패턴에서 크게 벗어나는 급증·급락**을 통계적 `SUSPECT` 후보로 표시한다.
- 용량 및 코어 수 변화는 **할당 이벤트일 수 있으므로 삭제하지 않는다**. 해당 변화가 있는 구간에서는 사용률 급변을 맥락과 함께 검토한다.

### 4.2 탐지 방법: 과거 관측값 기반 Hampel

각 VM에 대해 CPU 사용률과 메모리 사용률을 **각각** 검사한다.

1. VM별로 timestamp 순서의 **중복 제거 시계열**(§3.2)을 만든다. 동결본은 VM 안에서 이미 시간순이므로(역행 0건) 다시 정렬하지 않고, 역행이 발견되면 오류로 중단한다. 원본 timestamp의 숫자 자체는 바꾸지 않는다.
2. 해당 시점 **직전 13개 시점**(같은 구간 안)의 유효한 값으로 중앙값 `m`과 MAD를 계산한다(300초 간격이면 약 1시간 분량). 할당 0, `INVALID`, `duplicate_conflict`인 시점은 **건너뛴다**.
3. `robust_sigma = 1.4826 × MAD`로 정의한다.
4. `|현재 사용률 − m| > 3 × robust_sigma`이면 `SUSPECT`(`hampel_spike`)로 표시한다(양방향).
5. 직전 13개 시점 중 유효 관측치가 **7개 미만**이면 `insufficient_history`, `MAD = 0`이면 `zero_mad`로 **판정하지 않고** `UNDETERMINED`로 센다.
6. 간격이 **600초를 넘으면 새 구간**으로 보고 이력을 처음부터 다시 쌓는다(동결본 최대 간격 2,100초).

> **설정값 v1:** `window=13`, `min_valid=7`, `threshold=3`, `direction=both`, `segment_gap=600초`. 이 값들은 검증 전의 출발점이다. 최소 관측치와 임계값은 팀이 표본 VM을 확인해 변경할 수 있다. v1.2 실행 결과 `hampel_spike`는 CPU 743,415시점(전체 시점의 6.9%, 판정이 성립한 시점의 12.1%), Memory 635,835시점(5.9% / 7.3%)이다. 정상 급증이 과하게 잡히는지 표본 VM으로 확인한 뒤 임계값 상향을 검토한다.

> **짧은 VM 유의:** 동결기록 §5-2에 따르면 VM별 행 수는 **5행(vm 312) ~ 23,692행(vm 842)**이다. 행 수가 `min_valid` 이하인 VM은 통계 판정이 한 번도 성립하지 않는다. manifest 기준으로 그룹 A(7행 이하)는 **vm 312(5행)**, 그룹 B(12행 이하, §5.2)는 **vm 312·674·753(5·8·8행)**이 해당하며, 이 VM의 통계 검사 결과는 데이터 부족에 따른 `UNDETERMINED`가 된다. 이런 VM을 학습에서 제외할지(최소 관측 길이 하한)는 이상치 정책이 아닌 **별도 미결 사항**이다. **v1.2 실행 결과:** 정상/급증 판정이 한 번도 성립하지 않은 VM은 CPU 53개, Memory 13개다. 주된 사유는 짧은 시계열(CPU 3개·Memory 2개)보다 `zero_mad`(사용률이 일정함, CPU 42개·Memory 7개)와 `zero_allocation`(CPU 8개·Memory 4개)이며, VM별 내역은 `anomaly_summary.csv`에 있다.

### 4.3 실제 처리 기준

| 발견 사례 | 판정 | 처리 |
|---|---|---|
| CPU 20% → 95% → 22% | Hampel 후보일 수 있음 | **95% 보존**. 실제 부하 급증 가능성이 큼 |
| CPU가 90% 수준으로 2시간 지속 | 높은 부하지만 정상일 수 있음 | **전체 유지** |
| CPU 사용량이 `-500 MHz` | 명백한 오류 | **해당 값 `NaN`**, 판정 로그 |
| Memory 40% → 75%로 올라간 뒤 유지 | 메모리 확보·캐싱·워크로드 변경 가능 | **상승 구간 보존** |
| CPU cores 2 → 4 | 할당 상태 변화 가능 | **보존**. 이전·이후 상태 구분 기록 |
| CPU capacity 6,000 → 12,000 MHz | 할당 상태 변화 가능 | **보존**. 사용률 변동의 맥락으로 활용 |
| Memory 사용량이 capacity의 105% | 단위·수집 시점 문제 가능 | `SUSPECT`; **자동 삭제 금지** |

**그룹 A 확정 원칙:** `CPU cores`와 CPU/Memory capacity는 **유효성 검사만** 하고, Hampel은 **사용량을 용량으로 나눈 비율**에만 적용한다. Hampel에 걸린 값은 **원본 유지 + 플래그 기록**이 기본이다.

---

## 5. 그룹 B — Disk / Network 이상치 처리 정책

### 5.1 그룹 A와 다르게 다뤄야 하는 이유

Disk/Network는 **0이 오래 이어지고 짧은 순간에 매우 큰 값이 나타나는** 패턴이 정상적으로 발생할 수 있다. 단순 IQR, Z-score, 상위 1% 제거, 최대값 제한은 정상적인 I/O 버스트까지 삭제할 위험이 크다.

### 5.2 탐지 방법: 비음수 검증 + 보수적 시계열 후보 탐지

각 VM에서 다음 **4개 컬럼을 따로** 검사한다.

- Disk read throughput
- Disk write throughput
- Network received throughput
- Network transmitted throughput

**1차(실제 정제):** 음수 및 비유한 수(`inf`, `-inf`)만 `INVALID`로 처리한다. **0은 결측으로 바꾸지 않는다.**

**2차(분석용 후보 탐지):** 필요할 경우 다음 규칙으로 높은 급증만 `SUSPECT`로 표시한다.

1. 유효한 값 `x >= 0`에 대해 `y = log1p(x)`로 변환하여 분포의 치우침을 줄인다. **원본 `x` 자체는 변경하지 않는다.**
2. 해당 시점 **이전 관측값 25개**에서 `y`의 중앙값과 MAD를 구한다(약 2시간 분량).
3. `y > median(y_history) + 6 × 1.4826 × MAD(y_history)`일 때만 급증 후보로 표시한다(**상방 단측**).
4. 과거 유효 관측치가 **12개 미만**이거나 `MAD = 0`이면, 이 통계 기준만으로 판정하지 않는다(해당 짧은 VM은 §4.2 '짧은 VM 유의' 참조). 특히 0이 이어지다가 나타난 첫 양수 값을 임의로 삭제하지 않는다.
5. 급증 후보라 하더라도 **클리핑·삭제·절삭 평균 치환 금지**. 단순히 로그에 기록하고 필요한 경우 인접 시점 및 다른 I/O 지표를 살펴본다.

> **설정값 v1:** `window=25`, `min_valid=12`, `threshold=6`, `direction=upper_only`, `transform=log1p`. 전부 **보수적인 탐지 시작값**이지, 최적 검증된 절대 규칙은 아니다.

**v1.3 적용 규칙:**

- **판정 시계열:** Group A와 같은 중복 제거 시계열(§3.2)과 600초 구간 분리(§4.2-6)를 쓴다. 같은 시각 안에서 처리량이 서로 다르면 `UNDETERMINED`(`duplicate_conflict`)로 기록하고 그 시점은 이력에서 뺀다.
- **할당 0 시점(§3.1):** 처리량은 할당량과 무관하게 정의되므로 **판정에 포함**한다(이 절에 제외 규칙이 없음). 동결본 사전 확인에서 할당 0 행의 처리량은 96~99.7%가 0이었고, 제외하든 포함하든 급증 후보 수 차이는 지표당 14개 이하였다.
- **이벤트 표기:** `group = io`, `metric`은 해당 처리량 컬럼명, `original_value`는 `log1p` 변환 전의 원래 처리량(KB/s)이다.
- **사실(동결본 전수):** 4개 컬럼 모두 음수·inf·NaN이 **0건**이다. 0 비율은 Disk read 89.4%, Disk write 22.6%, Network received 54.1%, Network transmitted 51.2%이며, 모든 행이 0인 VM이 각각 254·15·65·110개다.
- **v1.3 실행 결과:** Group B `INVALID`는 **0건**이다. `hampel_spike`는 Disk read 66,325 / Disk write 586,513 / Network received 235,032 / Network transmitted 231,399시점이다(판정이 성립한 시점 대비 8.59% / 7.32% / 5.17% / 5.95%, 전체 시점 대비 0.62% / 5.45% / 2.18% / 2.15%). `zero_mad`가 가장 많아(Disk read는 전체 시점의 92.2%), 판정이 한 번도 성립하지 않은 VM이 Disk read 736개, Disk write 51개, Network received 259개, Network transmitted 238개다. Group A 부분 체크섬은 슬라이스 A 동결값과 일치했다.

### 5.3 실제 처리 기준

| 발견 사례 | 판정 | 처리 |
|---|---|---|
| Disk read 0 KB/s가 장시간 지속 | 정상 가능 | **0 유지** |
| Disk write가 순간적으로 30배 증가 | 정상 대량 쓰기 가능 | **값 유지**, 필요 시 `SUSPECT` |
| Network received가 급증 후 바로 0으로 복귀 | 정상 트래픽 버스트 가능 | **값 유지** |
| Disk/Network throughput이 음수 | 명백한 오류 | **해당 값 `NaN`**, 로그 기록 |
| 처리량이 매우 커 보이지만 장비 최대 대역폭을 모름 | 실제 처리량일 수 있음 | **상한을 임의로 설정하지 않음** |
| 특정 시점에 Disk write와 Network receive가 함께 급증 | 워크로드 변화 가능 | **서로 다른 지표를 참고**하되 자동 수정 금지 |

**그룹 B 확정 원칙:** **실질적인 값 변경은 음수·비유한 값 등 명백한 오류에 한정**한다. 통계적 급증 탐지는 분석·품질 진단용이며, `SUSPECT`도 **값 유지**가 기본이다.

---

## 6. 절삭 평균, IQR, Z-score, MAD의 역할 정리

| 기법 | 이번 단계에서의 역할 | 원본 값 변경에 사용? |
|---|---|---|
| **절삭 평균(Trimmed Mean)** | VM별 평상시 사용 수준을 요약하거나 정제 전·후 비교할 때 참고 | **아니오** |
| **IQR** | VM별 사용률 분포를 기초 진단할 때 참고 | **아니오** |
| **Z-score** | 평균·표준편차가 급증값에 민감하므로 주력 탐지기로는 비추천 | **아니오** |
| **MAD / Hampel** | 그룹 A의 변화 후보 탐지, 그룹 B의 보수적 급증 후보 탐지 | **아니오** (후보 표시만) |
| **유효성 규칙** | 음수 사용량·잘못된 용량·비유한 수 확인 | **예** (`NaN` 표시) |

**절삭 평균 권장 용도:** 예를 들어 VM별 CPU 사용률을 10% 절삭 평균으로 요약하면 대표적인 부하 수준을 비교할 수 있다. 다만 이를 **순간 급증값의 대체값**으로 사용하지 않는다. 과거 데이터를 기반으로 요약값을 만들 경우에도 향후 학습/검증 분할에서 미래 데이터가 섞이지 않도록 주의한다.

---

## 7. 구현 기준 — pandas / numpy

### 7.1 처리 순서

```text
data/vm_all_1250_loaded.csv (로딩 동결본, 읽기 전용)
    ↓
동결 체크섬 대조 + dtype 지정 읽기 + 기본 숫자형 검증
    ↓
vm_id 단위로 분리 (VM 내부는 원본 행 순서)
    ↓
공통 유효성 검사 (음수 / 비유한 값 / 유효하지 않은 용량)
    ↓
CPU·Memory: 사용률 계산 → 과거 윈도우 Hampel 후보 표시
Disk·Network: 비음수 검증 → log1p + 보수적 Hampel 후보 표시
    ↓
명백한 오류값만 NaN 표시 (행은 보존)
    ↓
data/vm_all_1250_clean.csv + 이상치 판정 로그 + 집계 보고서
```

**입력 읽기:** 동결기록 §2대로 `dtype={'Timestamp [ms]': 'Int64', 'CPU cores': 'Int64'}`를 지정한다. `CPU cores`는 nullable 정수(`Int64`)이므로 `INVALID` 처리한 셀은 `NaN`이 아니라 `<NA>`로 표시되며 dtype은 `Int64`로 유지된다.

**대용량 처리 원칙:** 입력은 **단일 병합 CSV** `data/vm_all_1250_loaded.csv`(1,168,186,561바이트 ≈ 1.09 GiB, 11,221,800행)이며, VM별 파일로 다시 쪼개지 않는다(로딩 설계안 rev.2 §3 결정). `src/load_data.py`가 1~1250번 파일을 번호순으로 이어 썼으므로 **같은 `vm_id`의 행은 연속 구간으로 저장**되어 있고, VM 내부는 원본 행 순서 그대로다. VM 단위 처리는 다음 중 하나로 한다.

- 전체를 읽은 뒤 `groupby('vm_id')`로 VM별 처리
- `chunksize`로 순회하며 `vm_id`가 바뀌는 경계에서 VM을 잘라 처리. 청크 끝에 걸린 VM의 행은 다음 청크로 넘겨 이어 붙인다.

한 VM은 최대 23,692행이므로 VM 하나를 통째로 메모리에 올려 창을 계산하면 청크 경계 문제가 생기지 않는다. 처리가 끝난 VM은 출력 파일에 바로 이어 쓰고, 메모리에서 전체를 다시 `concat`하지 않는다.

### 7.2 핵심 통계 판정 함수 예시

아래 코드는 **이미 시간순으로 정렬된 한 VM(또는 시간 연속 구간)의 시리즈**에 적용하는 *후보 탐지 함수*다. 자동 데이터 수정 함수가 아니다. 구현의 권위는 `src/clean_data.py`(벡터화·구간 분리·중복 제거 포함)이며, 아래는 개념 예시다.

```python
import numpy as np
import pandas as pd


def hampel_candidate(series, window, min_valid, threshold,
                     upper_only=False, log_transform=False):
    """현재 시점 이전 관측치만 이용해 이상치 후보(True)를 표시."""
    x = pd.to_numeric(series, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)

    # log1p는 0 이상의 I/O 데이터에만 사용한다.
    if log_transform:
        x = np.log1p(x.where(x >= 0))

    # shift(1)로 현재 시점이 자신의 기준값에 섞이지 않게 한다.
    history = x.shift(1)
    rolling = history.rolling(window=window, min_periods=min_valid)
    median = rolling.median()
    # 창 안의 NaN(할당 0·INVALID 등)은 건너뛴다. np.median을 쓰면 NaN 하나로 MAD 전체가 NaN이 된다.
    mad = rolling.apply(
        lambda values: np.nanmedian(np.abs(values - np.nanmedian(values))),
        raw=True,
    )
    sigma = 1.4826 * mad

    if upper_only:
        candidate = (x - median) > (threshold * sigma)
    else:
        candidate = (x - median).abs() > (threshold * sigma)

    # MAD=0일 때 통계적 임계값을 임의 생성하지 않는다.
    candidate = candidate & sigma.gt(0) & median.notna()
    return candidate.fillna(False)


# 적용 예시: CPU·Memory는 '사용률'을 임시로 계산
# df는 동일 VM의 연속된 시계열이 시간순으로 정렬된 DataFrame
cpu_pct = (
    df["CPU usage [MHZ]"]
    / df["CPU capacity provisioned [MHZ]"].where(
        df["CPU capacity provisioned [MHZ]"] > 0
    ) * 100
)
cpu_candidate = hampel_candidate(cpu_pct, 13, 7, 3)

memory_pct = (
    df["Memory usage [KB]"]
    / df["Memory capacity provisioned [KB]"].where(
        df["Memory capacity provisioned [KB]"] > 0
    ) * 100
)
memory_candidate = hampel_candidate(memory_pct, 13, 7, 3)

# Disk·Network은 각 컬럼에 동일 함수 적용
io_candidate = hampel_candidate(
    df["Disk write throughput [KB/s]"],
    window=25, min_valid=12, threshold=6,
    upper_only=True, log_transform=True,
)
```

> **성능 및 정확도 유의:** `rolling.apply()` 기반 MAD는 데이터가 많으면 비용이 높다. 하루 안에 처리할 때는 일부 VM으로 실행시간을 먼저 확인하고, 필요하면 탐지 대상을 줄이거나 구현을 최적화한다. 또한 **timestamp 간 누락 구간·중복·할당량 변화**가 존재하는지 먼저 확인하고, 불연속 구간은 분리하여 창을 계산한다. 위 코드는 그 전처리와 결과 저장을 생략한 핵심 로직 예시다.

### 7.3 정제 출력과 판정 로그

정제 CSV에는 **필수 원본 지표와 `vm_id`만 유지**한다. 이상치 정보는 별도 CSV로 저장한다.

```text
data/
├── fastStorage/2013-8/              # 원본 1,250개 CSV (수정 금지)
├── vm_all_1250_loaded.csv           # 로딩 동결본 (수정 금지, 정제 입력)
├── fastStorage_loading_manifest.csv # 로딩 검수 로그 (동결)
├── vm_all_1250_clean.csv            # 정제본: 명백한 오류만 NaN 처리한 단일 CSV
└── quality/
    ├── anomaly_events.csv           # INVALID·SUSPECT·duplicate_conflict 이벤트(롱 포맷)
    ├── anomaly_summary.csv          # VM별·지표별 판정 건수(롱 포맷)
    └── zero_allocation_runs.csv     # 할당 0 연속 구간(§3.1)
```

`anomaly_events.csv`의 권장 컬럼:

| 컬럼 | 의미 |
|---|---|
| `vm_id` | VM 식별자 |
| `timestamp_raw` | 원본 Timestamp 숫자 |
| `metric` | 해당 컬럼명 |
| `group` | `resource` / `io` / `common` |
| `status` | `INVALID` / `SUSPECT` / `UNDETERMINED` |
| `reason` | `negative_value`, `hampel_spike`, `over_capacity`, `zero_mad` 등 |
| `original_value` | 변경 전 관측값 |
| `action` | `set_nan` / `keep` |

**이벤트 기록 범위(v1.2):** `anomaly_events.csv`에는 `INVALID`, `SUSPECT`, `UNDETERMINED`(`duplicate_conflict`)만 행 단위로 남긴다. `VALID`와 나머지 `UNDETERMINED`(`zero_mad`, `insufficient_history`, `zero_allocation`, 시간 간격)는 수백만 건이므로 `anomaly_summary.csv`에 VM·지표별 **건수만** 기록한다.

- 이벤트 키는 `(vm_id, timestamp_raw, metric)`이며 중복 시각의 모든 행에 적용된다. 같은 키에서 사용량이 다른 행(값 충돌)은 사용량 값별로 한 줄씩 남긴다(할당량만 달라 충돌한 경우는 한 줄).
- Group A 판정의 `metric`은 사용량 컬럼명(`CPU usage [MHZ]`, `Memory usage [KB]`)이고 `original_value`는 그 사용량이다. 판정은 사용률로 하지만 사용률 자체는 저장하지 않는다(§2).

`anomaly_summary.csv` 컬럼: `vm_id`, `metric`, `group`, `status`, `reason`, `count`. `count`의 단위는 **시점 수**(중복 제거한 `(vm_id, timestamp)`)이며, 0건인 검사도 남겨 검사 실행 여부를 확인할 수 있게 한다. 시간 간격 검사는 `metric = Timestamp [ms]`, `group = common`으로 적고 인접 간격 단위로 센다(`regular_step` / `duplicate_timestamp` / `irregular_step` / `segment_break`). VM당 집계 행은 Group A 32행 + Group B 28행(4지표 × 7항목) = 60행이며, Group B 행은 VM마다 Group A 행 **뒤에** 붙인다(Group A 부분 체크섬 유지, `docs/GWA_T12_정제_슬라이스A_동결기록.md` §6). 이벤트도 같은 순서를 따른다.

`zero_allocation_runs.csv` 컬럼: `vm_id`, `resource`(`cpu`/`memory`), `start_timestamp_raw`, `end_timestamp_raw`, `rows`(파일 행 수, 중복 포함), `position`(`leading`/`middle`/`trailing`/`whole`), `rows_usage_positive`(할당 0인데 사용량 > 0인 행 수).

**동결본과 정제본 행 수가 동일한지**(둘 다 11,221,800행) 확인한다.

### 7.4 구현 슬라이스

| 슬라이스 | 범위 | 산출물 |
|---|---|---|
| **A (v1.2)** | 공통 시간 검사, 그룹 A 유효성·할당 0 구간·할당량 초과·Hampel, `CPU usage [%]` 일치 검사 | `src/clean_data.py`, `data/quality/` 3종 |
| **B (v1.3)** | 그룹 B 비음수·비유한 검사, `log1p` Hampel(상방 단측) | 같은 스크립트에 추가, `data/quality/` 재생성. Group A 부분 체크섬 유지 |
| 정제 CSV | B 완료 후 동결본에서 한 번에 생성 | `data/vm_all_1250_clean.csv` |

---

## 8. 작업 완료 기준 (1일 프로젝트용)

- [ ] 로딩 동결본(`data/vm_all_1250_loaded.csv`)은 변경하지 않고 정제본을 `data/vm_all_1250_clean.csv`로 별도 저장했다. 정제 후에도 동결본 SHA-256이 동결기록 §1과 일치한다.
- [ ] VM별 `CPU cores`·CPU/Memory capacity의 유효성을 검사했다.
- [ ] CPU·Memory 사용률로 그룹 A 이상치 후보를 검사했다.
- [ ] Disk·Network 4개 지표에서 0은 유지하고 음수/비유한 값만 오류 처리했다.
- [ ] Disk·Network의 급증은 삭제하지 않고 후보로 기록했다.
- [ ] 타임스탬프 원본을 보존하고, VM별 실제 기록 간격(300초 여부)과 중복·누락 간격을 별도 기록했다.
- [ ] 전체 원본에서 `CPU usage [%]` 중복 관계를 검증했다(스키마 제거는 이후 결정).
- [ ] `anomaly_events.csv`와 `anomaly_summary.csv`를 생성했다.
- [ ] **정제 전·후 행 수, 수정값 수, 최대 CPU·Memory 사용률, I/O 최고값을 비교**하여 실제 고부하가 훼손되지 않았는지 확인했다.

### 팀이 검토할 정책 변수

| 변수 | 시작값 | 조정 시 고려사항 |
|---|---|---|
| 그룹 A Hampel 윈도우 | 과거 13개 관측치 | 부하 변화가 빠르면 짧게, 패턴이 느리면 길게 |
| 그룹 A 임계값 | robust sigma의 3배 | 정상 급증 플래그가 많으면 상향 검토 |
| 그룹 B Hampel 윈도우 | 과거 25개 관측치 | I/O 버스트의 지속시간과 빈도 |
| 그룹 B 임계값 | robust sigma의 6배 | 희소한 I/O에서는 플래그를 최소화 |
| 절삭 평균 | 필요 시 양끝 10%씩 제외 | **요약 통계 전용**, 측정값 대체 금지 |

---

## 9. 최종 의사결정 한 장 요약

| 항목 | 그룹 A: CPU / Memory | 그룹 B: Disk / Network |
|---|---|---|
| 핵심 관점 | **할당량 대비 사용 상태** | **단위 시간당 I/O 처리량** |
| 1차 유효성 | 음수, 비유한 수; **할당 0은 상태로 유지** | 음수, 비유한 수; **0 허용** |
| 통계적 탐지 | 사용률 기준 Hampel (과거 13개, 3배, 양방향, 중복 제거·600초 초과 간격에서 구간 분리) | `log1p` + Hampel (과거 25개, 6배, 상방만) |
| 높은 값 처리 | CPU·Memory 고부하 **유지** | Disk·Network 버스트 **유지** |
| 할당량 초과 | 우선 `SUSPECT`, 즉시 삭제하지 않음 | 최대 전송률이 불명확하면 상한 임의 설정 안 함 |
| 확실한 오류 | 해당 값만 `NaN`, 행 보존 | 해당 값만 `NaN`, 행 보존 |
| 결과 저장 | 정제본 + 별도 이벤트·집계 로그 | 정제본 + 별도 이벤트·집계 로그 |

**한 줄 결론:** **CPU·Memory는 사용률 기반 Hampel로 변화 후보를 찾고, Disk·Network는 정상 버스트를 최대한 살리는 보수적 탐지만 적용한다. 모든 그룹에서 통계적 후보는 삭제하지 않고, 확실한 오류만 해당 값에 한해 처리한다.**

---

## 10. 개정 이력

| 버전 | 변경 내용 |
|---|---|
| v1.0 | 초기 정책안. VM별 CSV 로딩본·ZIP 압축 크기 전제. |
| v1.1 | `docs/GWA_T12_로딩_동결기록.md` 기준으로 정합화: ① 입력을 단일 병합 동결본 `data/vm_all_1250_loaded.csv`로, 출력을 `data/vm_all_1250_clean.csv`로 변경(VM별 `loaded/`·`cleaned/` 폐기) ② "128MB 압축 크기" 서술 삭제 → 실제 크기·행 수와 `vm_id` 연속 구간 처리 방식으로 교체 ③ Timestamp 초 단위 해석(§5-3) 반영, 300 = 300초 명시, 전체 간격 검증은 정제 첫 단계로 이관 ④ `CPU cores = 0` 실재(§5-1) 반영 ⑤ 짧은 VM(§5-2, manifest) 판정 불가 사례 명시 ⑥ 입력 dtype(`Int64`) 및 `<NA>` 표기 반영 ⑦ 완료 기준에 동결 체크섬 대조 추가 |
| v1.2 | 동결본 전수 사실 확인과 팀 결정을 반영해 Group A 정책 확정: ① 할당 0(cores·capacity = 0)을 `INVALID`에서 **상태(값 유지)**로 변경하고 구간 기록 추가(§3.1) ② 중복 timestamp는 중복 제거 시계열로 판정하고 행은 유지(§3.2) ③ 600초 초과 간격에서 구간 분리, 유효 관측치 기준 명확화(§4.2) ④ 판정 불가 건은 집계만, 이벤트는 INVALID·SUSPECT·duplicate_conflict만(§7.3) ⑤ `CPU usage [%]` 검사 규칙(0.01%p) ⑥ §7.2 예시의 MAD를 `np.nanmedian`으로 수정(창에 NaN이 있으면 판정이 누락되던 문제) ⑦ 구현 슬라이스 표 추가(§7.4) |
| **v1.3** | Group B 정책 확정: ① 중복 제거·구간 분리·값 충돌 규칙을 Group A와 공유 ② 할당 0 시점은 B 판정에 포함 ③ 이벤트 표기(`group = io`, 원래 처리량) ④ 집계 행 순서(VM마다 A 뒤에 B) ⑤ 동결본 사전 확인 사실과 실행 결과 기록(§5.2) |
