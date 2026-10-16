# T30 — T14 RF 파이프라인 요인 분리, 층화 주 평가 (사전등록 설계 초안)

**상태: 사전등록 확정 — 승인(2026-10-16, 제안 기본값)·구현 중·미실행.** (작성 당시: 설계 초안 — 미승인·미구현·미실행.) 이 문서에는 코드·설정·테스트·봉인·manifest·학습·실행 기록이 없다. 아래 수치는 모두 기존 감사 보고서·집계 파일·원장(메타데이터)에서 옮겼다. 작성 중 원시 CSV·OOF·모델 파일을 읽지 않았고 SHA도 새로 계산하지 않았다(기준 OOF SHA는 독립 검토자가 별도 재계산해 일치 확인). `earthquake/data/final_holdout/`은 읽거나 열거하지 않았다. §10 승인 항목이 확정되기 전에는 구현·`prepare`·학습을 하지 않는다.

**승인 기록(2026-10-16):** 사용자가 이 설계의 **제안 기본값**(Tier 1 4팔 + Tier 2 통제 C0·N1·N2 + Tier 3 DROP_TE_ALL·CW_NO_ROS = 완료 41 fit·시도 상한 45, 제안 임계값·실행 순서)으로 구현·실행을 승인했다. 승인 항목은 모두 문서의 제안값으로 확정하며, 결과를 본 뒤 기준·팔을 바꾸지 않는다. 위 상태 문단은 작성 당시 기록이다.

**가이드 대조 줄 (AGENTS 필수):** 과제 주제 가이드 **심화 항목**에 해당한다. 인용 문구 "전처리만 변경한 실험, 가중치만 변경한 실험, 샘플링만 적용한 실험을 분리하여 효과 비교"의 출처는 **과제 주제 가이드 원문 심화 항목(2026-10-16 사용자 제공 원문과 대조, 저장소 밖)**이다. 저장소 안 요약본 `docs/PROJECT_GUIDE_KO.md`에는 이 문장이 없고 §4 원칙 5·6("변경 요소를 분리하여 Macro F1 변화와 등급별 손익을 함께 기록")만 있다. `NO_ROS`·`CW_NO_ROS`는 필수 항목(불균형 처리 효과 비교: 클래스 가중치·재표집)과도 맞닿는다. 이 실험은 **주 평가(개발 Stratified 5-Fold seed 42) 결과를 낸다**. **판정에 쓰는 분할은 층화 seed 42 5-Fold 하나뿐**이며 geo2 Group 결과(T16)는 판정에 쓰지 않고 §7.5에서 인용만 한다. 개선 후보 채택 실험이 아니라 요인 분리 연구다.

- 제안 실험 ID: `T30_rf_factors_stratified_v1` (승인 시 확정). 브랜치는 `peter`만 쓰며 새 브랜치·worktree를 만들지 않는다.
- 제안 비공개 출력: `local_reproduction/T30_20261016_rf_factors_stratified/` (실행 전에 존재하지 않아야 함). 하위 `preflight/`, `run/`(manifest 사본 없음, 설계 봉인·원장·Fold 산출물·`backup/`), `audit/`, `logs/`.
- 제안 공개 보고: `reports/RF_FACTORS_STRATIFIED_<날짜>.md`와 집계 JSON. 독립 감사 PASS 후에만 만든다.
- 작성일 2026-10-16. 작성 시점 `peter` HEAD `192881772fba5dc904629fa19976c81c25669b5d`. 봉인할 때 그 시점의 전체 40자 커밋 SHA로 다시 결속한다.
- 다른 worker가 T31·T32 문서를 동시에 쓴다. 이 문서는 그 파일을 참조하거나 수정하지 않는다.

## 1. 질문과 범위

질문: 층화 주 평가에서 CV 1위인 T14 RF 파이프라인(RF + geo1/2/3 OOF Target Encoding + 지역 빈도 + 학습 Fold RandomOverSampler + 원시 age 유지)의 Macro F1 0.700427은 **어느 요소에서 오는가**.

- T14는 TE·ROS·RF·age995 원값 유지를 한꺼번에 가져온 재현이며 요인을 분리하지 않았다(`reports/TE_ROS_RF_REPRO_20261015.md` 결론 4).
- T16(`docs/T16_RF_FACTORS_PLAN.md`, `reports/RF_FACTORS_20261015.md`)이 같은 요인을 **geo2 Group 4분할에서만** 분리했다. 층화 값은 없다(`reports/STRATIFIED_SUMMARY_20261016.md` 표3). T30은 같은 요인 대조를 층화 5-Fold에 옮긴다.
- 기준선(T14 RF) 하나에 대해 **한 번에 한 요소만** 바꾼 팔을 비교한다. 조합 탐색·최적 팔 선정·튜닝·결정 규칙 변경·새 모델 계열은 범위 밖이다. 요인 효과를 더하거나 상호작용을 추정하지 않는다.

## 2. 기준선과 재사용

기준 팔 `T14_RF_TE_ROS_A`는 **감사 PASS인 T14 `run_v2`의 Stage A 층화 OOF를 재사용**한다. 기준 RF 재학습은 0회다(§3 통제 C0의 1회 점검 제외, 승인 항목 3).

| 항목 | 값 | 근거 |
|---|---|---|
| 기준 OOF | `local_reproduction/T14_20261015_te_ros_rf_repro/run_v2/strat/RF_TE_ROS/oof.csv.gz` | T14 보고·집계 JSON |
| 기준 OOF SHA-256 | `69a74d410da08bf06c9da35211e3fe6f1e424bd4afbc72c4de2d2d3c21c09969` | `reports/te_ros_rf_repro_20261015.json` `per_scheme.strat.oof_sha256` |
| T14 설계 SHA-256 | `965f8741dd48ea0b277a9a512b551fd05ac1f365bd9a4d11fd9769688db09a74` (사전등록 `b23a374`, 수정1 `813fab3`) | T14 보고, T16 `config.json` 핀 |
| 분할 | `split_manifest.csv` SHA `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721`, Fold당 학습 166,784/검증 41,696 | `configs/data_contract.json`, 층화 종합 §2 |
| 기준 성능 | Fold F1 0.700261/0.701455/0.700451/0.701229/0.698739, 평균 **0.700427**(SD 0.001070, ddof=1), Accuracy 0.739884, Recall3 0.673016, 3→1 744, 3→2 22,071 | T14 집계 |
| 기준 풀링 혼동행렬(행=실제) | `[[12435,7252,412],[6898,94857,16852],[744,22071,46959]]` | T14 보고 Stage A |

재사용 확인 방법(구현 후 `preflight`·`prepare`·`verify`·`finalize`가 각각 반복):
1. 파일 SHA-256이 위 값과 같다.
2. `t14.reference(path, sha, ids, y, folds)`와 같은 방식으로 `building_id` 정렬 후 건물 ID·Fold 배열·라벨이 `load_development()`와 manifest 순서에 일치한다.
3. `design.json`이 SHA `965f8741…`이고 그 `config`·`source_sha256`이 현재 T14 `config()`·`sources()`와 같다(T16 `baseline_design(c)`가 `c`의 `baseline_run`·`baseline_design_sha256`만 쓰므로 그대로 재사용 가능).
4. 기준 행렬 재구성이 T14 `build_fold_matrices`와 **비트 동일**하다(Fold 0 `Ao`/`Bo`/`Xos`/`yos`; T16 `preflight`가 쓰는 방식).

재사용 가능 판정: **가능.** T14 Stage A는 원본 혼동행렬과 셀 단위로 일치(EXACT)했고 독립 감사 PASS이며, 확률·예측은 저장돼 있다. 한계는 기준 값이 층화 seed 42 5-Fold에서 학습된 고유 설정 56개(층화 종합 §4 순위 표 하단 문구: 60행 중 T15 사후 규칙 파생 3행과 T0 중복 T1 A 1행 제외) 가운데 1위라는 **선택 통계**라는 점이다(§9).

## 3. 팔 정의 — 기준 대비 한 요소씩

공통 고정: 원시 입력 38개 보존, raw geo 숫자 3열 유지, 문자 8열 기존 범주 스키마 one-hot, geo1/2/3별 등급1·3 TE 및 빈도 3열, m=20, 내부 TE는 층화 분할에 한해 **원본 `StratifiedKFold(5, shuffle=True, random_state=0)`**(T14 Stage A와 동일, `inner_splits_for('strat')`가 `None`을 반환), outer-train에만 ROS(`random_state=0`), age 원시값, RF 100 trees/`random_state=0`/`n_jobs=4`와 sklearn 1.8.0 나머지 기본값, 예측은 단일 스레드 `argmax+1`. 환경은 T14·T16과 같은 동결 환경(Python 3.13.5, sklearn 1.8.0 등)이다.

### 3.1 Tier 1 — 핵심 4팔 (T16과 같은 정의)

| arm | 기준 대비 바꾸는 요인 | 가이드 항목 대응 | 그대로 두는 것 | 모델 열 수(T16 관측) |
|---|---|---|---|---|
| `ONEHOT` | geo TE 6열을 outer-train에서 학습한 geo1/2/3 one-hot 블록으로 **대체**(raw geo 숫자·빈도 3열 유지) | 전처리 | ROS, age, RF | 77−6+(geo 어휘 수); T16 10,262~10,372 |
| `NO_ROS` | ROS를 끈다 | **샘플링만** | 모든 전처리 열·값, age, RF | 77 |
| `DROP_GEO23_TE` | geo2·geo3 TE 4열만 제거(geo1 TE·빈도·raw geo 유지) | 전처리 | ROS, age, RF | 73 |
| `AGE_C` | age995를 C 처리(outer-train 비995 중앙값 대체 + `age_is_995` 표시) | 전처리 | geo 인코딩/빈도, ROS, RF | 78 |

- `ONEHOT`은 TE 블록 **대체** 효과이지 전체 지역 처리의 효과가 아니다. `AGE_C`는 대체값·표시 변수 한 묶음이며 각각을 분리하지 않는다. 열 수가 달라지는 팔은 `max_features='sqrt'`가 후보 특성 수도 바꾼다. 따라서 고정 RF에 표현 변경을 적용한 조건부 비교이며 순수 정보량 효과가 아니다(T16과 같은 해석 제한).
- Tier 1만으로는 TE **전체**(6열)의 기여와 **가중치만** 바꾼 효과가 비어 있다. 아래 Tier 3이 그 공백을 채운다.

### 3.2 Tier 2 — 통제 (팔이 아니라 해석을 위한 대조, 권고)

| 통제 | 내용 | fit |
|---|---|---:|
| `C0_BASE_REFIT` | 새 실행기로 **기준 파이프라인을 Fold 0에서 1회 재학습**해 T14 OOF Fold 0 확률과 비교. 같은 매트릭스·같은 RF 설정이 새 실행기에서 같은 확률을 내는지 확인하는 동일성 게이트 | 1 |
| `N1_RS1`, `N2_RS2` | 기준 파이프라인에서 **RF `random_state`만** 1, 2로 바꾼 5-Fold 재학습. 팔 간 Δ가 RF 난수 재추첨만으로 얼마나 흔들리는지(잡음 바닥)를 측정 | 5+5 |

- C0 통과 조건: 모든 예측 라벨 일치와 최대 확률 차 ≤ 1e-12(정확 일치 예상; 1e-12는 T14 재로딩 허용 오차). T14 OOF CSV는 반드시 `float_precision='round_trip'`(T14 `read_pred`와 같은 정밀 파서)로 읽는다. 기본 파서는 상위 두 확률이 거의 같은 행에서 1ulp 차이로 argmax를 바꿔 T14 감사 1차 시도의 오류 원인이 됐다. 상위 두 확률이 정확히 같은 동률 행(T14 Fold 1에 237건 관측)은 `argmax`의 첫 인덱스 규칙을 양쪽에 같게 적용하며, 확률 비트 일치를 우선 기준으로 삼고 라벨 불일치는 해당 행 확률 차가 정확히 0인 동률일 때를 제외하고 실패로 본다. 불일치하면 **전체 중단**하고 원인 기록(설계 개정 사안이며 임의 재시도 아님).
- N1/N2는 팔 판정의 합격 조건이 아니라 **주석용 잡음 지표**다(§7). 행렬 구성은 기준과 같으므로 `fit` 시간은 기준과 비슷하다.

### 3.3 Tier 3 — 확장 2팔 (선택, 승인 항목 4)

| arm | 내용 | 가이드 항목 대응 |
|---|---|---|
| `DROP_TE_ALL` | geo TE 6열을 모두 제거(raw geo 숫자·빈도 3열 유지, ROS·age·RF 동일). Tier 1의 `ONEHOT`(대체)·`DROP_GEO23_TE`(부분 제거)와 달리 TE **전체의 기여**를 한 번에 본다 | 전처리 |
| `CW_NO_ROS` | ROS를 끄고 RF `class_weight='balanced'`만 추가(그 외 `NO_ROS`와 동일). `NO_ROS` 대비 **가중치만**, 기준 대비 "ROS → 가중치" 교체를 본다 | **가중치만** |

- 두 팔은 Tier 1과 같은 기준에 대한 비교이며 Tier 1 판정에 영향을 주지 않는다. `CW_NO_ROS`는 기준 대비 두 요소(ROS 제거, 가중치 추가)가 다르므로 **`NO_ROS` 대비 한 요소**로만 해석한다(표에 두 대조를 모두 표시, 기준 대비 단일 요인으로 쓰지 않는다. 판정은 §7.4).
- 가중치 방식은 `class_weight='balanced'`(전체 학습 행 기준)로 고정한다. 트리별 `balanced_subsample`은 쓰지 않는다. 변경 필요 시 새 설계다.

### 3.4 T0 RF 최고와의 관계 — 참조표만 (신규 학습 없음, 승인 항목 10)

T0 개인 비교의 RF 최고(`class_weight=balanced`, `max_features=0.2`, `min_samples_leaf=1`, v1.0.1 one-hot ≈1.26만 열)는 CV Macro F1 0.658667(±0.000974), Accuracy 0.685932, Recall3 0.679866, 3→1 1,516, 3→2 20,821이다(`reports/STRATIFIED_SUMMARY_20261016.md` 표1). T14의 0.700427(3→1 744, Recall3 0.673016)과는 **적어도** 다음이 함께 다르다: 전처리(one-hot vs TE+빈도+raw geo), 불균형 처리(class_weight vs ROS), `max_features`(0.2 vs sqrt), `max_depth`(T0 초기 20; 튜닝 후 유지 여부 미확인 vs T14 기본 None). 따라서 **한 요소 비교가 아니며** T30 판정에 쓰지 않는다. 보고서에는 위 T0 값과 T30 팔 값을 같은 표에 **기술 참조**로만 나란히 두고 "차이를 설명했다"고 쓰지 않는다. T0 RF와의 완전한 가교(복수 요소를 단계적으로 바꾸는 별도 사다리 실험)는 이번 범위 밖의 후속 선택이다.

## 4. 기존 실행기 재사용 판단

T16 `experiments/rf_factors/run.py`(봉인 소스, 수정 불가)를 읽고 판단했다.

| 구성 | 재사용 | 이유 |
|---|---|---|
| `build_fold_matrices(F_train, X_train, y_train, F_valid, scheme, c, arm)` | **가능(import)** | `scheme`에 의존하는 부분은 `t14.inner_splits_for(scheme, …)`뿐이며 `'strat'`이면 `(None, None)`을 반환해 원본 내부 Fold를 쓴다. 검증 라벨은 인자가 아니다. 단 `c['arm_changes']`에 신규 팔(`DROP_TE_ALL`, `CW_NO_ROS`)을 지원하는 키가 없다 |
| `apply_state`, `matrix_equal` | 가능 | 스킴 비의존 |
| `t14.load_inputs`, `original_frame`, `fit_rf`, `predict_proba_deterministic`, `reference`, `compare_arms`, `pooled_scores`, `write_pred`/`read_pred`, `read_ledger`, `check_saved`, `atomic_write` | 가능 | 모두 스킴 비의존. `load_inputs`의 `old_fold`가 층화 manifest Fold이고 데이터 계약이 `split_manifest.csv` SHA를 검증한다 |
| `baseline_design(c)` | 가능 | `c`의 `baseline_run`·`baseline_design_sha256`만 사용 |
| `config()` | **불가** | `c['schemes'] == SCHEMES`(geo2 4개)·80 fit·84 시도·T14 geo 기준 OOF를 하드코딩 |
| `input_bundle`, `baseline_check`, `preflight`, `prepare`, `verify`, `finalize`, `targets` | **불가** | geo manifest를 읽고 `t5.validate_assignment(…, GEO[:2], …)`로 **지역 그룹 비중첩**을 요구한다. 층화 manifest는 그룹이 Fold에 걸치므로 검증이 실패한다. 기준 OOF도 geo2 4개로 고정 |
| `fit_fold`, `fit_all` | 사실상 불가 | 로직은 거의 스킴 비의존이나 모듈 상수 `STUDY_ID='T16_RF_FACTORS'`·`targets()`·`ARMS`·`judge()`(20 Fold/4분할 가정)에 묶여 있다. 상수를 덮어쓰는 방식은 기록의 `study_id`를 오염시키므로 쓰지 않는다 |

**결론: T16 실행기를 그대로 층화에 쓸 수는 없다.** 필요한 범위는 얇은 신규 실행기 `experiments/rf_factors_strat/`(`run.py`, `config.json`, `__init__.py`)와 `tests/test_rf_factors_strat.py`다.
- 위 "가능" 항목은 T16/T14 모듈에서 import해 **재사용**한다(두 모듈의 소스 파일 SHA를 봉인 소스 목록에 넣는다).
- 신규 코드: 층화 전용 `config()`(스킴 `['strat']`, 단일 기준 OOF 핀, fit·시도 상한, 임계), `targets()`, `prepare/verify/preflight`(그룹 비중첩 검증 제거·`load_inputs`의 층화 Fold 사용), `fit_fold`·`fit_all`(T16 로직을 새 `STUDY_ID`로 복사), `assemble/finalize`(5-Fold 판정, §7).
- 신규 팔 지원은 T16 빌더를 그대로 부를 수 없다.
  - T16 `build_fold_matrices`는 첫 줄에서 `require(arm in ARMS or arm == BASE, 'Unknown arm')`로 알려진 팔만 받는다. `ARMS`는 T16 모듈 상수(4팔)라서 `DROP_TE_ALL`·`CW_NO_ROS`·`N1/N2`·`C0`를 넘기면 거부된다. 상수를 덮어쓰는 것은 봉인 소스 위반이므로 하지 않는다. 신규 팔은 빌더를 `BASE`(기준) 이름으로 호출한 결과에 대한 후처리 또는 RF 파라미터 변경으로 구현한다.
  - `DROP_TE_ALL`은 TE 열 이름 규칙 `{geo}_te0/_te2` 6열을 `Ao`·`Bo`·`Xos`에서 제거하되, **저장되는 `state['columns']`·`state['selected']`·`state['drop']`도 같은 열 집합으로 동시에 필터**한다. `apply_state`가 `state['columns']`로 검증 행렬을 재구성하므로 상태만 옛 열을 가지면 재로딩 검증이 실패하거나 열이 어긋난다. ROS가 열에 독립이므로 열 필터 뒤에도 행·라벨 정렬이 같은지 테스트한다.
  - `CW_NO_ROS`·`N1/N2`·`C0`는 RF 파라미터(`class_weight`, `random_state`)만 팔별로 달리한다. 이 팔별 RF 파라미터는 **봉인되는 config 사본의 `arm_rf_params`에 명시**하고(예: `CW_NO_ROS: {class_weight: balanced}`, `N1_RS1: {random_state: 1}`), 실행 시 `fit_rf`는 이 사본만 읽는다. `n_jobs=4`와 나머지 기본값은 공통이며, 모델의 `get_params()`가 봉인된 값과 같은지 Fold마다 검사한다.
  - T16 코드 수정은 없다.
- 구현 후 `tests/`에 합성 계약 테스트(층화 manifest 허용, 그룹 비중첩 검증 없음, 팔별 열 집합 차이, 상한·재시도 규칙)를 둔다. 추가로 (a) **복사한 `fit_fold`/`fit_all`이 같은 합성 입력에서 T16 원본과 동일하게 동작**하는지(저장 산출물·원장 키·재개·재시도 거부 규칙·SHA 검증이 `study_id` 문자열만 다르고 같은 결과) 비교 테스트, (b) 신규 `config()`가 T16과 같은 **환경 핀 검사**(`c['environment'] == t14.environment()`)와 `rf_params`·`te` 스냅샷 동일성 검사를 수행하는지, (c) `DROP_TE_ALL`의 `state['columns']` 동기화와 재로딩 비트 동일성 테스트를 포함한다. 실제 개발 데이터·홀드아웃·기존 모델을 읽는 테스트는 만들지 않는다.

## 5. fit 수·시도 상한·재시도·순서·병렬도·예상 시간

### 5.1 fit 수와 상한

| 구성 | 신규 완료 fit | 시도 상한(+재시도 4) |
|---|---:|---:|
| Tier 1 (4팔 × 5 Fold) | 20 | 24 |
| + Tier 2 (C0 1 + N1 5 + N2 5) | 31 | 35 |
| + Tier 3 (2팔 × 5 Fold) | 41 | 45 |

- 제안 기본값은 **Tier 1+2+3 = 41 fit, 시도 상한 45**(승인 항목 3·4·6). 사용자가 Tier를 줄이면 위 표의 해당 줄을 쓴다. 봉인 후 변경하지 않는다.
- 기준 RF 재학습은 C0의 1회뿐(Tier 2 미승인 시 0회). 튜닝·early stopping·결정 규칙 변경·자동 재시도는 없다. 홀드아웃 접근 없음.

### 5.2 재시도 규칙 (T16과 동일)

- 인프라 중단(프로세스 사망 등)만 `--retry-interrupted --retry-reason "<사유>"`로 **수동** 재시도한다. 같은 봉인 설계로 Fold(타깃)당 최대 1회, 전체 최대 4회 추가. 성능을 본 뒤 파라미터를 바꾸는 용도로 쓰지 않는다. 알고리즘·데이터 결함은 중단 후 새 개정 설계로 다룬다.
- 완료 Fold는 원본·백업 SHA를 검증하고 재학습하지 않는다. 실패 시도의 원장·로그·부분 파일은 `.incomplete_attempt<N>`로 보존한다. 프로세스 사망 후 남은 `RUNNING.lock`은 실제 실행 주체가 없는지 컨트롤 타워가 확인한 뒤에만 제거한다. T16에서 프로세스가 기록 없이 사라진 선례가 있으므로(`reports/RF_FACTORS_20261015.md`), 실행 전에 절전·자동 잠금·세션 종료 설정을 실행 관리자가 확인한다. 이 문서는 예약 실행이나 백그라운드 서비스를 설정하지 않는다.

### 5.3 실행 순서와 병렬도

- 순서(고정): `C0`(Fold 0) → `N1`·`N2`(Fold 0~4) → `NO_ROS`·`DROP_GEO23_TE`·`AGE_C`·`DROP_TE_ALL`·`CW_NO_ROS`(Fold 0~4) → `ONEHOT`(Fold 0~4). 저비용 항목을 먼저 끝내 부분 결과를 확보하고 C0 게이트를 맨 앞에 둔다. ONEHOT은 마지막이다.
- 병렬도: **RF 학습 내부 `n_jobs=4` 한 단계뿐.** 팔·Fold·통제는 직렬이고 외부 병렬은 없다(`outer_parallel_jobs=1`). 한 번에 한 주체만 학습한다. 희소 행렬을 전체 `.toarray()`로 변환하지 않는다(ONEHOT은 CSR 유지).

### 5.4 예상 시간 (모두 미측정 — 기존 기록 기반 추정)

| 항목 | 근거(실측 기록) | 추정 |
|---|---|---|
| 기준형 팔 1 fit(C0, N1/N2, AGE_C, DROP_GEO23_TE, DROP_TE_ALL, CW_NO_ROS) | T14 층화 Fold fit 9.9~11.5초(평균 약 10.4초), TE 약 0.4초·ROS 약 0.2초(Fold 0). T16 geo2: AGE_C 8.7초, DROP_GEO23_TE 8.7초 | fit 약 6~12초 + 저장·재로딩·SHA·백업 포함 Fold당 수십 초 |
| `NO_ROS` 1 fit | T16 geo2 평균 5.3초 | fit 약 5초대 |
| `ONEHOT` 1 fit | T16 geo2 20 Fold fit 합계 10,704.7초 = 평균 535.2초(최대 642.4초). 로그상 Fold 간 벽시계 간격 약 600~660초(저장·재로딩·백업 포함), 모델 파일 약 195 MiB | fit 약 450~700초, Fold 벽시계 약 10~13분. 층화에서는 학습에 모든 geo3 코드가 더 많이 보여 열 수가 약간 늘 수 있어 상향 가능 |

- ONEHOT 5 Fold ≈ **약 50~65분**, 저비용 36 fit ≈ **약 10~20분**, `preflight`(Fold 0 전처리만, 분류기 학습 0) 수 분. 합계 **약 1.3~1.8시간**(Tier 1만이면 ONEHOT 비중으로 약 1.1~1.4시간)을 잠정 예상한다. 이는 완료 시간 보장이 아니며 첫 ONEHOT Fold의 실측 값으로 갱신한다. 감사·보고 작성은 별도다.
- 디스크: `prepare` 시 여유 60 GiB, 각 Fold 전 15 GiB를 요구한다(T16과 같은 값; 승인 항목 8). ONEHOT 모델은 원본+백업으로 5 Fold 합계 약 2 GiB(195 MiB × 5 × 2) 안팎으로 예상한다(미측정). 나머지 팔·통제 모델은 소형이라 합계 수백 MiB 이내로 보며, 전체 출력(원본+백업+OOF CSV+로그)은 약 3~4 GiB로 추정한다(미측정). 요구 여유 60/15 GiB는 이 추정보다 훨씬 보수적이다.

## 6. 보고 지표

각 팔·통제를 기준 OOF와 같은 Fold로 짝지어 다음을 모두 기록한다(주지표 labels=[1,2,3] Macro F1).
- Fold 5개의 Macro F1, Fold별 Δ(팔−기준), 평균 Δ, **양의 Fold 수/5(Fold 승수)**, ΔSD(ddof=1).
- 풀링 OOF(208,480행 1회씩)의 클래스별 Precision/Recall/F1·지지 수, 혼동행렬, Accuracy, Recall3, **3→1·3→2 건수와 비율**(분모는 실제 3등급 69,774건), 예측 비중, 기준 대비 예측 전환(`experiments.geo_ablation.audit.turnover`와 같은 정의).
- 특성 수, fit 시간, RF 불순도 중요도(지역 블록 합산; 인과 해석 금지), ROS 후 클래스 건수.
- 풀링 Macro F1과 Fold 평균 Macro F1은 서로 다른 요약임을 표에 구분 표기한다. 표본 CV가 아니라 전체 개발 CV다.

## 7. 판정 — 요인 분리의 기술 판정 (개선 채택 아님)

팔은 "T14에서 한 요소를 바꾼 결과"이므로 **Δ<0은 기준에 포함된 요소가 층화 CV에 기여했다는 기술 관측**, Δ>0은 그 요소가 오히려 손해라는 관측일 뿐이다. 최고 팔 선정, 팔 간 우열, 효과 합산, 최종 모델 승격은 사전등록된 판정 밖이다. 임계값은 **제안값이며 승인 항목 5**다. 봉인 후 결과를 보고 바꾸지 않는다.

### 7.1 Macro F1 라벨 (팔별 1개)

기준 Fold SD 0.00107과 T12·T14·T16 선례의 효과 임계 0.003을 따른다.

| 라벨 | 조건 |
|---|---|
| `CHANGE_IMPROVES` | 평균 Δ ≥ +0.003 이고 양의 Fold ≥ 4/5 |
| `CHANGE_WORSENS` | 평균 Δ ≤ −0.003 이고 음의 Fold ≥ 4/5 (= 기준의 해당 요소가 기여) |
| `NEGLIGIBLE` | \|평균 Δ\| < 0.003 |
| `MIXED` | \|평균 Δ\| ≥ 0.003 이나 같은 부호 Fold가 4/5 미만 |

### 7.2 오류 프로파일 플래그 (라벨과 독립, 팔별 병기)

- `RECALL3_DOWN`/`RECALL3_UP`: 풀링 Recall3 Δ가 ≤ −0.01 / ≥ +0.01.
- `TO1_UP`/`TO1_DOWN`: 풀링 3→1 건수가 기준 744 대비 +10% 이상(≥ +75건) / −10% 이상(≤ −75건). 3→2는 건수·비율을 보고하되 플래그는 두지 않는다.
- 라벨이 `CHANGE_IMPROVES`여도 `RECALL3_DOWN`·`TO1_UP`이 함께 있으면 "손익 상쇄"로 병기한다(층화에서도 가중치·재표집이 3→1을 늘린 선례, 층화 종합 §3). 안전성 개선·악화는 이 플래그를 바탕으로 서술하되 3→1 상한은 승인 대상 오류 비용 판단이다(T29 초안 승인 항목 9와 같은 사안).

### 7.3 잡음 주석 (Tier 2 승인 시)

N1·N2의 Fold별 Δ(RS1/RS2 − 기준)에서 `max |평균 Δ|`와 `max |Fold Δ|`를 보고한다. `max |평균 Δ| ≥ 0.0015` 또는 `max |Fold Δ| ≥ 0.003`이면 보고서의 모든 라벨에 `주석: RF 난수 재추첨 잡음이 임계와 같은 규모` 를 붙인다(라벨 자체는 바꾸지 않음). 이 값은 5 Fold·난수 2개 기준의 거친 값이며 신뢰구간이나 p-value가 아니다.

**부호 있는 Δ도 함께 보고한다.** N1/N2의 Fold별·평균 Δ(RS − 기준)를 절댓값 요약과 별도로 부호 그대로 표에 싣는다. 기준 `random_state=0`은 56개 설정 중 최고값으로 선택된 값이므로, 같은 파이프라인을 다른 난수로 돌렸을 때 평균 Δ가 체계적으로 음수이면 이는 **승자 편향(선택 통계)의 직접 측정치**다(§9). 이 음의 Δ를 팔 Δ에서 빼서 보정하지 않는다(보정은 사후 분석이며 사전등록 판정 밖).

**한계: 잡음의 하한이다.** N1/N2는 RF `random_state`만 바꾸고 ROS(`random_state=0`)와 내부 TE Fold(`StratifiedKFold(5, shuffle, 0)`)는 고정한다. 따라서 ROS 재추첨·내부 TE 분할 변동이 더해진 실제 잡음은 이보다 클 수 있으며, 측정값은 RF 난수 한 요인의 하한으로만 읽는다. ROS·내부 TE Fold 시드 변동은 이번에 측정하지 않는다.

### 7.4 `CW_NO_ROS`의 판정 (Tier 3, `NO_ROS` 대비)

`CW_NO_ROS`는 기준 대비 두 요소가 달라 §7.1~7.2를 기준에 직접 적용하지 않는다. **비교 대상은 `NO_ROS`**(같은 Fold·같은 매트릭스에서 ROS 없음·가중치 없음)이며 아래를 별도로 적용한다. 기준 대비 값은 참고로만 병기한다.
- 라벨: §7.1과 같은 임계(평균 Δ ±0.003, 같은 부호 Fold ≥4/5)를 `Δ = CW_NO_ROS − NO_ROS`에 적용한다. 표에 "NO_ROS 대비"를 명시한다.
- 플래그: `RECALL3_UP/DOWN`은 `NO_ROS`의 풀링 Recall3 대비 ±0.01. `TO1_UP/DOWN`의 **기준 건수는 744가 아니라 `NO_ROS` 실측 3→1 건수**이며 상대 ±10%를 `finalize`에서 계산한다. `NO_ROS`의 3→1 건수가 작아 ±10%가 10건 미만이면 절대 ±10건 이상일 때만 플래그를 세운다(보조 임계, 승인 항목 14).
- `NO_ROS` 자체는 기준(ROS 있음) 대비로 §7.1~7.2를 적용하므로 3→1 기준 건수가 비교마다 다르다(기준 744, `CW_NO_ROS` 비교에서는 `NO_ROS` 값). 보고서 표 머리에 기준 건수를 숫자로 적는다.

### 7.5 geo2 결과는 판정에서 제외, 인용만

아래 T16 값은 `reports/RF_FACTORS_20261015.md`에서 옮긴 geo2 Group 20 Fold(4 seed×5) 결과이며 **판정에 쓰지 않는다**. 보고서에는 층화 값과 나란히 두고 "방향이 같은가/다른가"를 기술 표기만 한다(지역 의존성 심화 해석은 `docs/RESEARCH_STATUS.md` 평가 기준 절의 geo2 심화 항목 범위).

| 팔 | T16 geo2 평균 Δ | 양의 Fold | 풀링 Recall3 Δ |
|---|---:|---:|---:|
| ONEHOT | +0.012231 | 13/20 | −0.098020 |
| NO_ROS | −0.000182 | 10/20 | −0.018392 |
| DROP_GEO23_TE | −0.005238 | 4/20 | +0.015941 |
| AGE_C | −0.002799 | 7/20 | +0.005772 |

층화와 geo2는 학습 때 본 지역/못 본 지역이라는 서로 다른 질문이다. 두 결과가 다르면 "분할에 따라 요인 기여가 다르다"는 관측으로만 서술하고 하나를 주로 격상하거나 평균 내지 않는다.

## 8. 체크포인트·원장·SHA·재개·감사·보고 절차

AGENTS 운영 순서(로컬 보존 → 실험별 체크포인트 → 독립 감사 → GitHub 보고)를 따른다. 명령 진입점은 구현·봉인 후 `docs/RESEARCH_STATUS.md`에 기록하며 이 문서는 실행 명령을 확정하지 않는다.

1. **구현 승인 → 사전등록 커밋:** `experiments/rf_factors_strat/`·테스트·이 계획의 최종본을 커밋하고 봉인 소스 SHA 목록(T14 소스 + T16 소스 + 신규)을 설정에 넣는다. 전체 `python -m unittest discover -s tests -v`를 실행해 pass/skip을 구분해 기록한다.
2. **`preflight`(분류기 학습 0):** 데이터 계약·층화 manifest SHA, 기준 OOF 재사용 검증(§2의 1~4), Fold 0에서 팔별 행렬 구성과 인코더 상태 재생성, 기대 열 수(기준 77·NO_ROS 77·DROP_GEO23_TE 73·AGE_C 78·DROP_TE_ALL 71·ONEHOT은 기록만), 기준 행렬 T14 비트 동일성.
3. **`prepare`:** 소스가 커밋된 HEAD와 일치하고 clean한지 확인하고 `design.json`/`design_seal.json`(설계 SHA)·빈 `ledger.json`·`backup/`을 만든다. 출력 폴더가 이미 있으면 거부한다. 봉인 후 코드·설정·계획은 변경하지 않는다.
4. **`fit-all`(단일 실행권):** `RUNNING.lock`을 배타 생성해 두 실행자가 동시에 들어가지 못하게 한다. 각 Fold 직후 모델·인코더 상태·17자리 확률 CSV·Fold JSON을 저장하고 SHA-256과 상태를 원장에 기록한다. 저장 직후 (a) 단일 스레드 재로딩 확률의 비트 동일성, (b) CSV round-trip, (c) 별도 `backup/` 복사 SHA를 검증한 뒤에만 완료로 표시한다. 실패하면 완료로 표시하지 않고 중단한다. 완료 Fold는 재학습하지 않으며 중단 후 `--retry-interrupted` 규칙으로 재개할 수 있다. 원본 산출물은 덮어쓰지 않는다.
5. **`status`:** 완료/시도/실행 중 Fold 수를 확인한다.
6. **`finalize`:** 팔별 OOF를 조립하고(중단 후 재개 가능, 기존 값이 다르면 실패) 기준 OOF와의 Δ·라벨·플래그를 계산해 `analysis.json`을 쓴다. 완료 fit 수가 승인 구성과 다르면 최종화를 거부한다.
7. **독립 감사:** 실행기를 import하지 않고 별도 스크립트로 재계산한다. 확인 항목은 (a) 설계 봉인·소스·환경·manifest·기준 OOF SHA, (b) 원장 완료 수·시도 수 ≤ 상한·재시도가 사유 규칙 안, (c) 산출물 원본·백업 SHA, (d) Fold 멤버십·라벨·`y_pred=argmax`, (e) 저장 모델을 읽어 검증 예측 재추론 확률 일치(감사 중 RF 재학습 0), (f) outer-train 라벨만으로 TE 맵·prior·빈도·ONEHOT 어휘·age 중앙값 재계산 일치, (g) 팔이 기준과 의도한 한 요소만 다른지(열 이름 집합 차이: NO_ROS 없음/DROP_GEO23_TE −4/DROP_TE_ALL −6/AGE_C +1·`age_is_995`/ONEHOT TE 6열 제외·one-hot 추가, `NO_ROS`·`CW_NO_ROS`의 `fit_rows`=학습 행 수), (h) C0 동일성, (i) Δ·Fold 승수·라벨·플래그·N1/N2 잡음 주석 재계산, (j) 홀드아웃 경로 접근 기록 없음. 감사 전에는 보고서를 결과 확정으로 커밋하지 않는다.
8. **공개 보고:** 감사 PASS 후 컨트롤 타워가 집계만 검토해 `reports/`에 보고서와 집계 JSON을 넣고 `docs/RESEARCH_STATUS.md`·`docs/PLAN.md`를 갱신한다. 행별 예측·확률·모델·OOF·manifest·자격증명은 공개 Git에 올리지 않는다. 커밋·푸시는 메인이 조율한다.

## 9. 홀드아웃과 한계

- **홀드아웃(`final_holdout`)에 접근하지 않는다.** 경로 가드(`require_not_holdout`)를 그대로 쓴다. 과거 홀드아웃은 T0 최종 모델 평가에 1회 쓰였고 후속 선택·튜닝·재평가에 재사용하지 않는다. T30 결과는 새 독립 성능이 아니다.
- **seed 42 층화 Fold의 반복 노출:** 같은 5개 Fold로 수십 개 설정을 이미 비교했다. 기준 0.700427은 그중 최고값인 **선택 통계**이며 승자 편향이 있다. 따라서 팔의 Δ가 음수로 치우칠 수 있고(기준이 운 좋은 쪽으로 뽑혔을 가능성), Fold를 짝지어도 이 편향이 완전히 상쇄되지 않는다. 이를 이유로 §7의 라벨을 "층화 5-Fold 기술 판정"이라 부르고 일반화 주장을 하지 않는다.
- **표본 크기:** 판정 단위가 5 Fold(건물은 한 번씩만 예측)이다. Fold는 독립 표본이 아니며 p-value·신뢰구간·인과 효과를 주장하지 않는다. 양의 Fold 4/5 기준은 부호 일관성의 거친 점검일 뿐 유의성이 아니다.
- **RF 난수 고정:** 팔은 `random_state=0` 하나만 쓰고 N1/N2는 기준에만 둔다. 팔별 난수 변동 폭은 측정하지 않는다(Tier 2 미승인이면 잡음 바닥도 없음).
- **범주 어휘의 전체 개발 데이터 사용:** 기준 파이프라인(T14 `original_frame`)은 문자 8열을 `pd.Categorical`로 만들어 어휘를 **개발 전체 값**에서 얻는다(라벨 무관 스키마). AGENTS "전체 데이터에 인코더/스케일러를 fit하지 않는다"를 엄격히 읽으면 이 어휘 고정이 해당하는지 해석 문제가 있으며, T29 초안 §4가 다루는 쟁점과 같다. 모든 팔과 기준이 같은 `original_frame`을 쓰므로 **팔 간 Δ에는 영향이 없다고 판단**하지만 절대 성능(0.700427)에는 영향 가능성이 남는다. 어휘는 라벨을 쓰지 않으며 TE·geo one-hot·age 중앙값·ROS는 outer-train에서만 fit한다. 이번에 이 파이프라인을 바꾸지 않고 한계로 기록하며 처리 방침은 승인 항목 13이다.
- **분할 한정:** 학습 때 본 지역 분포(층화)에서의 요인 기여이다. 못 본 지역 일반화는 T16 geo2가 다루며 층화와 다른 질문이다. 다른 층화 seed(101/202/303 등)는 이번에 쓰지 않는다. 필요하면 T29의 새 manifest 결정 후 별도 설계한다.
- **ONEHOT의 TE 대체 해석과 열 수/`max_features` 연동**, **AGE_C의 묶음 처리**, **요인 상호작용 미측정**, **`CW_NO_ROS`의 이중 변경**(§3.3)은 해당 팔의 해석 제한이다.
- **재현 한계:** 실제 fit 내부 행렬은 저장하지 않으므로 학습 호출의 적합성은 봉인 소스·사전 점검·원장·재추론 일치에 의존한다(T16 감사와 같은 한계).
- 3→1 오류는 요인 분리 결과와 함께 **항상 병기**한다. 개선처럼 보이는 F1 변화가 3등급 과소예측 증가를 동반하는지 본다.

## 10. 승인 항목

1. 실험 ID `T30_rf_factors_stratified_v1`·비공개 출력 `local_reproduction/T30_20261016_rf_factors_stratified/`·신규 모듈 `experiments/rf_factors_strat/` 구현 승인(T14/T16 봉인 소스는 수정하지 않고 import만 함).
2. Tier 1 핵심 4팔(`ONEHOT`, `NO_ROS`, `DROP_GEO23_TE`, `AGE_C`)의 정의 확정과 가이드 대조 줄(심화 항목, 층화 주 평가 결과, 판정 분할=층화 seed 42 5-Fold).
3. Tier 2 통제 포함 여부: `C0_BASE_REFIT`(기준 1 fit 재학습 게이트)와 `N1_RS1`/`N2_RS2`(RF 난수 잡음 10 fit). 권고: 포함.
4. Tier 3 확장 2팔(`DROP_TE_ALL`, `CW_NO_ROS`) 포함 여부와 가중치 방식(`class_weight='balanced'`). 권고: 포함(TE 전체 기여와 "가중치만" 효과를 보려면 필요).
5. 판정 임계 제안값: ΔMacro F1 ±0.003, 같은 부호 Fold ≥4/5, Recall3 ±0.01, 3→1 ±10%, 잡음 주석 기준(평균 0.0015·Fold 0.003). 3→1 상한은 오류 비용 판단이므로 판정 규칙이 아니라 병기 플래그로 둔다.
6. fit·시도 상한: Tier 구성에 따라 20/24, 31/35, 41/45 중 선택. 재시도는 인프라 중단 한정·Fold당 1회·전체 4회.
7. 실행 순서(저비용 먼저, C0 게이트 선행, ONEHOT 마지막)와 병렬도(RF `n_jobs=4` 한 단계, 외부 직렬), 단일 실행 주체 지정.
8. 디스크 요구(`prepare` 60 GiB·Fold 전 15 GiB)와 ONEHOT 모델 저장(원본+백업).
9. 공개 보고 범위: 집계 MD/JSON, geo2(T16) 인용표 포함, 오류·혼동행렬 집계 포함 여부.
10. T0 RF 최고(0.658667)는 신규 학습 없이 기술 참조표로만 병기할지, 별도 가교 실험을 후속으로 둘지.
11. 추가 층화 seed 반복은 이번에 하지 않고 T29 manifest 결정 후 다룰지.
12. 독립 감사 담당 지정(실행 관리자와 다른 주체)과 보고 시점.
13. 문자 8열 `pd.Categorical` 어휘가 개발 전체에서 오는 점(§9)을 라벨 무관 스키마로 허용하고 한계 기록으로 충분한지, outer-train 어휘로 바꾸는 새 설계가 필요한지(T29 §4와 같은 결정). 권고: 팔 간 동일하므로 허용·한계 기록.
14. `CW_NO_ROS`의 `NO_ROS` 대비 판정 규칙(§7.4)과 보조 임계(절대 ±10건).

## 11. 작성 시 확인하지 못한 사항 (미확인)

- 신규 팔 `DROP_TE_ALL`의 구현 방식(후처리 열 제거)이 T16 빌더와 열 순서·dtype에서 비트 동일한지는 구현 때 단위 테스트로 확인해야 한다.
- 층화 ONEHOT의 정확한 열 수와 fit 시간(미측정), 모델 파일 크기(T16 geo2에서 약 195 MiB 관측).
- T0 RF 최고의 `max_depth`·`n_estimators` 정확값(`BASELINE_REPORT.md`는 초기 100 trees/depth 20/leaf 2/sqrt만 기록, 튜닝 후 유지 여부 미확인).
- (확인됨) 기준 OOF `.../run_v2/strat/RF_TE_ROS/oof.csv.gz`의 실제 SHA가 `69a74d41…09969`임을 독립 검토자가 재계산해 일치 확인했다. 이 문서 작성자는 계산하지 않았으며 `preflight`가 구현 후 다시 확인한다.
- 현재 작업 트리에 다른 세션의 미커밋 변경이 있을 수 있어 봉인 시점 clean 여부는 그때 다시 확인한다.
