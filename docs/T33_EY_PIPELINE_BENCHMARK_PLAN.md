# T33 — 팀원 EY 노트북 파이프라인 층화 CV 재현 벤치마크 사전등록 설계

**상태: 사전등록 확정 — 승인(2026-10-16, 제안 기본값)·구현 중·미실행 (작성 당시: 설계 초안 — 미승인·미구현·미실행).** 이 문서에는 코드·설정·테스트·봉인·학습·실행 명령이 하나도 없다. 노트북 숫자는 노트북에 저장된 출력과 셀 소스를 옮긴 것이고, 비교군 숫자는 감사된 보고서·집계에서 옮겼다. 작성 중 원시 CSV·행별 OOF·모델 내용을 읽지 않았다(비교군 OOF는 파일 SHA-256 계산만 했고 내용은 열지 않았다. 비공개 `attempts.jsonl`·`analysis.json`의 집계 필드만 읽었다). `earthquake/data/`는 열거하지 않았고 `earthquake/data/final_holdout/`은 접근하지 않았다. §15 승인 항목이 확정되기 전에는 구현·`prepare`·학습을 하지 않는다.

**승인 기록(2026-10-16):** 사용자가 이 설계의 **제안 기본값**(E1 충실 재현·E2 ES 분리판 2팔, 완료 20 fit·시도 상한 24, D1~D4 제안 임계값, 노트북 설정 고정·재탐색 금지)으로 구현·실행을 승인했다. 결과를 본 뒤 기준·팔을 바꾸지 않는다. 노트북 원본과 팀원 브랜치는 수정·접근하지 않는다.

> **가이드 대조 줄 (AGENTS 필수):** 이 실험은 **팀원 파이프라인 재현 벤치마크**다. 공통 프로젝트 가이드의 "모든 팀원이 전체 분석·모델링을 수행하고 결과를 비교·통합" 항목(팀원 결과 비교·통합)과 주제 가이드의 심화 성격(추가 피처·추가 모델)에 해당하며, 필수 네 모델(LR·DT·RF·sklearn GradientBoostingClassifier)·불균형 처리 필수 항목을 **대체하지 않는다**. 주 평가(개발 Stratified 5-Fold seed 42, `split_manifest.csv`)의 층화 결과를 내고, **판정에 쓰는 분할은 층화 seed 42 5-Fold 하나뿐**이다. geo2/geo1 Group 분할은 이 실험에서 실행하지 않는다.

- 실험 ID: `T33_ey_pipeline_benchmark_v1`. 브랜치는 `peter`만 쓰며 새 브랜치나 worktree를 만들지 않는다.
- 비공개 출력(실행 전에 존재하지 않아야 함): `local_reproduction/T33_20261016_ey_pipeline/`. 하위: `source_snapshot/`, `preflight/`, `run/`(설계 봉인·원장·Fold 산출물·`backup/`), `audit/`, `logs/`.
- 제안 코드(승인 후 구현): `experiments/ey_pipeline_benchmark/`(`config.json`, `run.py`), `tests/test_ey_pipeline_benchmark.py`(합성 데이터만). 공개 보고 후보: `reports/EY_PIPELINE_BENCHMARK_<날짜>.md`와 집계 JSON(독립 감사 PASS 뒤에만 작성).
- 작성일 2026-10-16. 제출·병합 마감은 2026-10-19 23:59다. 이 문서는 T30~T32 계획과 담당 파일이 다르다. 학습 실행은 한 번에 한 주체만 하므로, 다른 학습 실험과 동시에 돌리지 않는다.

## 0. 요약

1. 대상은 팀원 EY의 노트북 `earthquake/data/지진.ipynb`(SHA-256 `480d63cd6adc3f68f70142ff159d229654db5d17a27d3af093baa38b831ac07b`, 사용자가 받아 둔 사본)의 셀 25 파이프라인이다. 원본 38개 입력 + 파생변수·지역 구성·조합별 피해 비율을 더한 **75개 변수** + LightGBM(노트북이 고른 설정 1개)이다.
2. 노트북의 결과(테스트 Macro F1 0.7143)는 **260,601행 전체의 단일 80/20 분할** 값이다. 우리 개발 자료(208,480행)의 층화 5-Fold와 분할·행 구성·학습 크기가 다르므로 **직접 비교할 수 없다**. 이 실험은 같은 파이프라인을 우리 Fold에서 다시 돌려 기존 후보(T8·T14·LR-C·T0)와 **같은 Fold, 같은 지표**로 놓는 것이 목적이다.
3. 팔은 둘이다. **E1 충실 재현**(노트북 순서 그대로)과 **E2 ES 분리판**(학습 Fold 안에서 tr/val을 먼저 나누고 early stopping 단계의 특징을 tr 안에서만 계산). E2는 라벨 혼입 제거 외에 ES용 val 특징 생성 방식(OOF 대 조회)과 ES 단계 지역 표의 기준 행 수(약 166k 대 133k)도 함께 바꾸므로, E1과 E2의 차이를 단일 원인에 귀속하지 않는다(§3.4). 하이퍼파라미터는 재탐색하지 않고 노트북이 고른 설정으로 고정한다.
4. 핵심 관찰(§3.4): 노트북 순서의 검증 라벨 혼입(tr 특징에 val 라벨이 섞임)은 **내부 검증 점수와 early stopping 트리 수**에만 영향을 준다. 최종 재학습 특징은 학습 Fold 전체로 계산한 것이라 E1과 E2가 **동일**하다. 따라서 두 팔의 바깥 Fold 점수는 모두 바깥 검증 행의 라벨을 어디에도 쓰지 않으며, 두 팔의 바깥 성능 차이는 트리 수(`n_trees`)에서만 생긴다. 다만 그 트리 수 차이는 ES 프로토콜 전체(라벨 혼입, val 특징 생성 방식, 지역 표 기준 행 수)의 합이다.
5. 신규 학습: **20 fit**(Fold 5 × 팔 2 × [early stopping fit 1 + 전체 재학습 1]), 시도 상한 24 fit. 학습 시간은 **미측정**이며 거친 추정 약 0.5~2시간이다(§5).
6. 판정은 채택이 아니라 **사전 고정 기술 판정**이다(§8). 3→1 오류, Recall3, 등급별 P/R/F1을 함께 보고하고, 노트북 단일 분할 값과의 차이는 §9의 원칙으로만 해석한다. 공식 개인 최종 모델(T0)은 이 실험으로 바뀌지 않는다.

## 1. 이 실험의 성격 (먼저 읽기)

- **재현 벤치마크이며 평가가 아니다.** 팀원 작업물의 품질을 평가·순위화하는 것이 아니라, 팀원 파이프라인을 우리 공통 Fold에서 다시 계산해 비교 가능한 숫자를 얻는 것이다. 보고서는 "EY 노트북 파이프라인의 층화 CV 재현 벤치마크"로 표기하고, 오류·결함이라는 단정 표현을 쓰지 않는다. 노트북에서 관찰한 검증 순서의 라벨 혼입 가능성은 E1 대 E2의 측정으로만 다루며, 그 차이는 ES 프로토콜 차이의 합으로 기술한다.
- **노트북은 읽기 전용이다.** 원본 파일(`earthquake/` 아래)은 수정하지 않고, 팀원 브랜치(EY·jjjjin·sy)에도 접근하지 않는다. 노트북 사본은 **비공개 폴더에 SHA를 고정해 보존**하고, 우리 실행기는 그 논리를 우리 코드로 옮겨 구현한다.
- **하이퍼파라미터를 새로 찾지 않는다.** 노트북이 8개 조합 중 고른 설정 하나를 그대로 쓴다. 우리 Fold 점수를 보고 설정을 바꾸지 않는다.
- 새 방법·임계값 조정·결정 규칙 변경·홀드아웃 평가·최종 모델 변경은 없다. 예측은 확률 argmax다.
- 팀원 파이프라인이 우리 후보보다 높게 나오더라도 곧바로 최종 후보가 되지 않는다. 후보 승격(예: T29 확인 평가 후보 추가)은 사용자 결정이다(§15).

## 2. 노트북에서 직접 확인한 사실

노트북 JSON을 UTF-8로 열어 셀 소스와 저장된 출력을 읽었다(셀 29개, nbformat 4). 환경 기록은 커널 `earthquake (3.14.6)`뿐이다.

| 항목 | 확인 내용 |
|---|---|
| 핵심 셀 | 셀 22(쉬운 버전: 지역·재료·유형 파생, 66번 줄 이후 실행부), 셀 23(기존 대 새 파생변수 XGBoost 비교, 출력 있음), **셀 25(여러 모델 튜닝 비교, 최종 파이프라인, 출력 있음)**. 셀 22·25의 `execution_count`는 `null`이고 출력은 남아 있다. 셀 23과 25의 함수 본문(`add_building_features`, `add_one_level`, `add_target_rates` 등)은 줄 단위 대조에서 동일하고, 셀 22는 키 4개·`min_count` 인자가 없는 이전 판이다. |
| 입력 | `nepal_building_damage.csv`(UTF-8-sig)를 새로 읽는다(셀 25). 셀 11의 희소 부차 용도 6열 병합·삭제는 EDA용 `df`에만 적용되고 **셀 25 파이프라인은 CSV를 다시 읽어 38개 입력을 모두 쓴다**(`ORIGINAL` = `building_id`·`damage_grade`를 뺀 전 열). 특수값 처리(age 995 포함)는 하지 않는다. 즉 **노트북이 지우는 열은 없다**. 셀 4 출력으로 확인한 원본은 260,601행 × 40열이며 열 순서는 우리 `FEATURES` 순서와 같다. |
| 분할 | 전체 260,601행에서 `train_test_split(test_size=0.2, stratify=damage_grade, random_state=42)` → 학습 208,480 / 테스트 52,121(셀 24 `test.shape` = (52121, 89)). 학습 208,480행은 우리 개발 행 수와 같지만 **행 구성이 같다는 근거는 없다**(우리 `make_splits`는 `building_id`로 정렬한 뒤 분할). 튜닝은 학습 안에서 다시 `train_test_split(test_size=0.2, stratify, random_state=42)`로 tr/val을 나눈다. |
| 변수 75개 | 원본 38 + `material_group`·`structure_type` + `n_weak_material`·`n_material` + 지역 구성 12(g1/g2/g3 × 흙·돌 비율, RC 비율, 지역 중앙 연령 대비 차이, 건물 수) + 조합 키별 3등급·1등급 비율 16(`old_keys` 4개 + `geo_keys` 4개, 각 high/low) + 새 5(`weak_age`, g2/g3 재료 high/low gap 4). 키 `key_material`의 비율 2개는 gap 계산에만 쓰이고 모델 입력이 아니다. 범주 10열(원본 문자 8 + `material_group`·`structure_type`)은 학습 데이터 기준 `pd.Categorical`로 LightGBM에 넘긴다(native categorical). **지역 ID 3열은 정수 그대로 수치 입력**이다. |
| 지역 구성 | 학습 데이터로만 계산. 건물 20채 미만 지역은 전체 값(`MIN_BUILDINGS`). 테스트·검증 행은 학습 표로 매핑하고 미관측 지역은 전체 값/0. |
| target rate | 조합 키별 3등급·1등급 비율. 학습 행은 `KFold(5, shuffle=True, random_state=42)` OOF, 테스트 행은 학습 전체 표. `old_keys` 최소 50채, `geo_keys` 최소 10채, `key_material` 최소 50채. 작은 조합은 전체 비율로 대체. |
| 가중치 | `(n / (3 · count_c)) ** strength`(클래스별 개수 기준). 최고 설정은 strength 0.3. |
| LightGBM 최고 설정 | `weight 0.3, learning_rate 0.03, num_leaves 31, min_child_samples 20, subsample 1.0, colsample_bytree 0.5`, `subsample_freq=1`, `random_state=42`, `n_jobs=-1`, `verbose=-1`. `n_estimators`는 최대 2000, early stopping 50(검증 `multi_logloss`, 가중치 없음), 재학습은 `best_iteration_`을 `n_estimators`로 고정. 검증 Macro F1 0.7068, 테스트 Macro F1 0.7143, 3개 재현율(1/2/3등급) 0.6764/0.8043/0.6740, 시험 8개 조합, 소요 약 2.66분(8조합 + 재학습, 하드웨어 미상). |
| **트리 수** | 최고 설정의 `best_iteration_` 값은 **노트북 어디에도 출력되지 않는다**(전 셀 출력에서 `n_trees`·`best_iter` 검색 결과 없음). 따라서 "노트북 출력값으로 트리 수 고정"은 불가능하다. |
| 최종 모델 선택 | 5개 모델의 **테스트 점수 내림차순 정렬**로 1등을 고른다(LightGBM 0.7143, XGBoost 0.7131, CatBoost 0.7123, 랜덤포레스트 0.7109, 로지스틱 0.6951; 1~2위 차이 0.0012). 가이드의 "검증으로 선택 후 Test 1회"와 순서가 다르다. 단 각 모델 설정 자체는 검증 점수로 골랐다. |
| 기준선 | 기존 파생변수 70개 XGBoost(500 rounds, lr 0.1, depth 6, 가중치 없음) 테스트 0.7020, 새 5개 추가 75개 0.7028. |
| 실행 이력 혼재 | `execution_count`가 불연속(1~32, null 혼재)이고 셀 27이 `NameError: train`을 낸다. 한 커널에서 위에서 아래로 재실행한 이력이 아니며 출력이 여러 세션에서 섞여 있다. 같은 파일의 셀 23(0.7028)과 셀 25의 기준선 표기(0.7028)는 일치한다. |
| 라이브러리 | 셀 25 출력에 `LGBMDeprecationWarning: The argument 'eval_set' is deprecated, use 'eval_X' and 'eval_y'`가 있다. 우리 동결 환경의 `lightgbm 4.6.0`에는 `eval_X` 인자가 없으므로(소스 확인) **노트북은 더 새로운 LightGBM으로 실행됐다**. 정확한 버전과 pandas·numpy 버전은 파일에 없다(`DataFrame ... str` dtype 표기로 pandas 3 계열로 추정, 미확인). |

## 3. 설계

### 3.1 공통 고정 설정 (두 팔 모두, 재탐색 금지)

| 항목 | 값 |
|---|---|
| 자료·분할 | `load_development()`가 돌려주는 개발 208,480행과 저장된 `split_manifest.csv`(SHA `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721`) 5-Fold. Fold당 학습 166,784 / 검증 41,696. 개발 CSV는 `development.csv` SHA `8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1`. |
| 학습 Fold 안의 계산 | 지역 구성, target rate, 범주 어휘, 가중치, tr/val 분할을 **각 학습 Fold 안에서만** 계산한다. 바깥 검증 Fold 행은 변환(조회)만 한다. 전체 개발 자료에 어떤 통계도 fit하지 않는다. |
| 모델 | `LGBMClassifier`, multiclass. `learning_rate 0.03`, `num_leaves 31`, `min_child_samples 20`, `subsample 1.0`, `subsample_freq 1`, `colsample_bytree 0.5`, `random_state 42`, 나머지는 LightGBM 4.6.0 기본값(`max_depth −1`, `reg_alpha/lambda 0`, `max_bin 255` 등). 범주 파라미터(`cat_smooth` 등)는 기본값을 그대로 쓰고 모델 텍스트의 실효값을 Fold별로 기록한다. |
| 가중치 | strength 0.3, 클래스 개수는 **해당 단계의 학습 행**(early stopping 단계는 tr, 재학습 단계는 학습 Fold 전체)에서 센다. 검증 점수·early stopping 지표에는 가중치를 쓰지 않는다(노트북과 동일). |
| early stopping | 최대 2000 rounds, patience 50, 지표는 기본 `multi_logloss`(검증 = 해당 단계의 val). 상한 2000에 닿으면 `cap_hit`을 기록하고 설정은 바꾸지 않는다. |
| 결정 규칙 | `predict_proba`의 argmax(1/2/3등급). 임계값·보정 없음. |
| 재현성 설정 | `deterministic=true`, `force_col_wise=true`(T8 관행, 감사 재추론용). 노트북에는 없다(§3.5 차이 6). |
| 지역 ID | 노트북처럼 **정수로 되돌려 수치 입력**한다. 우리 `load_development()`는 문자열이므로 정수로 변환하고 `str(int(x)) == x`가 전 행에서 성립함을 `prepare`에서 검사한다. 이는 파이프라인 충실 재현을 위한 것이며 지역 코드 크기에 물리적 의미가 있다는 주장이 아니다(§13). |
| 이진 열 | `uint8`을 `int64`로 되돌려 합산 파생이 노트북과 같은 정수 연산이 되게 한다(오버플로 방지). |
| **열 순서 (75개, 고정)** | 노트북 셀 25의 `ALL_FEATURES = FEATURES + NEW_FEATURES` 구성 순서를 직접 읽어 적은 것이다. `colsample_bytree=0.5`는 열 순서·위치에 의존해 열을 뽑으므로 순서가 결과를 바꾼다. 모든 특징 구성(A·B)과 감사 행렬이 이 순서를 따라야 하며 `prepare`가 일치를 검사한다. 구성 = `ORIGINAL`(38, CSV 열 순서) + `NEW_CAT`(2) + `['n_weak_material','n_material']`(2) + `REGION_COLS`(12) + `OLD_RATES`(8) + `GEO_RATES`(8) + `NEW_FEATURES`(5). 범주(pandas Categorical)로 넘기는 10열은 8·9·10·11·12·13·14·26·39·40번이다. |

**75개 열 정확한 순서** (`번호:이름`)

| 구간 | 열 |
|---|---|
| ORIGINAL 1~38 | 1:`geo_level_1_id` 2:`geo_level_2_id` 3:`geo_level_3_id` 4:`count_floors_pre_eq` 5:`age` 6:`area_percentage` 7:`height_percentage` 8:`land_surface_condition` 9:`foundation_type` 10:`roof_type` 11:`ground_floor_type` 12:`other_floor_type` 13:`position` 14:`plan_configuration` 15:`has_superstructure_adobe_mud` 16:`has_superstructure_mud_mortar_stone` 17:`has_superstructure_stone_flag` 18:`has_superstructure_cement_mortar_stone` 19:`has_superstructure_mud_mortar_brick` 20:`has_superstructure_cement_mortar_brick` 21:`has_superstructure_timber` 22:`has_superstructure_bamboo` 23:`has_superstructure_rc_non_engineered` 24:`has_superstructure_rc_engineered` 25:`has_superstructure_other` 26:`legal_ownership_status` 27:`count_families` 28:`has_secondary_use` 29:`has_secondary_use_agriculture` 30:`has_secondary_use_hotel` 31:`has_secondary_use_rental` 32:`has_secondary_use_institution` 33:`has_secondary_use_school` 34:`has_secondary_use_industry` 35:`has_secondary_use_health_post` 36:`has_secondary_use_gov_office` 37:`has_secondary_use_use_police` 38:`has_secondary_use_other` |
| NEW_CAT 39~40 | 39:`material_group` 40:`structure_type` |
| 재료 개수 41~42 | 41:`n_weak_material` 42:`n_material` |
| REGION_COLS 43~54 | 43:`g1_weak_ratio` 44:`g1_rc_ratio` 45:`g1_age_vs_region` 46:`g1_n_buildings` 47:`g2_weak_ratio` 48:`g2_rc_ratio` 49:`g2_age_vs_region` 50:`g2_n_buildings` 51:`g3_weak_ratio` 52:`g3_rc_ratio` 53:`g3_age_vs_region` 54:`g3_n_buildings` |
| OLD_RATES 55~62 | 55:`region_material_high_rate` 56:`region_material_low_rate` 57:`region_structure_high_rate` 58:`region_structure_low_rate` 59:`material_structure_high_rate` 60:`material_structure_low_rate` 61:`region_material_floor_high_rate` 62:`region_material_floor_low_rate` |
| GEO_RATES 63~70 | 63:`g2_high_rate` 64:`g2_low_rate` 65:`g3_high_rate` 66:`g3_low_rate` 67:`g2_material_high_rate` 68:`g2_material_low_rate` 69:`g3_material_high_rate` 70:`g3_material_low_rate` |
| NEW_FEATURES 71~75 | 71:`weak_age` 72:`g2_material_high_gap` 73:`g2_material_low_gap` 74:`g3_material_high_gap` 75:`g3_material_low_gap` |

(`key_material` 비율 `material_high_rate`·`material_low_rate`는 gap 계산 중간값이며 모델 입력이 아니다. `key_*` 문자열 열과 `floor_band`도 입력이 아니다. 목록은 위 구성식을 동결 환경에서 실행해 75개·중복 0임을 확인했다.)

### 3.2 E1 — 충실 재현 (노트북 순서)

학습 Fold마다 다음을 순서대로 수행한다.

1. **특징 구성 A**: 학습 Fold 전체를 노트북의 `train`, 바깥 검증 Fold를 `test`로 보고 `add_building_features` → `add_region_composition` → `add_target_rates`(old_keys, geo_keys, key_material) → gap·`weak_age` 순서로 변수 75개를 만든다. 학습 행의 target rate는 학습 Fold 전체에 대한 `KFold(5, shuffle=True, random_state=42)` OOF, 바깥 검증 행은 학습 Fold 전체 표다.
2. 특징 구성 A의 학습 행을 `train_test_split(test_size=0.2, stratify, random_state=42)`로 **tr / val**로 나눈다(학습 Fold 행을 `building_id` 순으로 놓은 위치 기준. E2와 같은 tr/val 구성을 쓴다).
3. **early stopping fit**: tr(특징 구성 A의 tr 행)로 학습하고 val로 중단하여 `best_iteration_`(= `n_trees`)과 내부 검증 Macro F1을 기록한다.
4. **재학습 fit**: 학습 Fold 전체(특징 구성 A 전체)로 `n_estimators = n_trees`로 학습한다.
5. 바깥 검증 Fold 확률을 저장한다.

### 3.3 E2 — ES 분리판 (early stopping 단계 특징을 tr 안에서만 계산)

1. 학습 Fold의 **원시 행**을 먼저 E1과 같은 방식으로 tr / val로 나눈다.
2. **특징 구성 B**: tr을 `train`, val을 `test`로 보고 노트북과 같은 함수로 변수 75개를 만든다(지역 구성·target rate·범주 어휘·가중치를 tr 안에서만 계산, **val은 tr 전체 표로 조회**). tr 행의 target rate는 tr 안의 `KFold(5, shuffle=True, random_state=42)` OOF다. 즉 지역 표는 약 133k행(tr)에서 만들어지고 E1의 약 166k행(학습 Fold 전체)과 기준 행 수가 다르다.
3. **early stopping fit**: 특징 구성 B의 tr로 학습, val로 중단하여 `n_trees`와 내부 검증 Macro F1을 기록한다.
4. **재학습 fit**: 학습 Fold 전체를 `train`으로 한 **특징 구성 A**(E1과 동일한 행렬)로 `n_estimators = n_trees`로 학습한다.
5. 바깥 검증 Fold 확률을 저장한다.

### 3.4 E1과 E2의 차이는 무엇인가 (설계의 핵심 해석)

- 노트북은 target rate를 학습 전체에서 OOF로 만든 **뒤** tr/val을 나눈다. 그래서 tr 행의 특징에 val 행의 라벨이 섞인다(val 라벨로 계산된 이웃 Fold 표가 tr 행 값에 들어간다). 이 때문에 **내부 검증 점수는 낙관적일 수 있고**, 8개 조합 선택과 early stopping 트리 수는 그 점수에 의존한다.
- 그러나 **최종 재학습과 테스트(바깥 검증) 특징은 학습 Fold 전체로 만든 것**이라 바깥 검증 라벨을 쓰지 않는다. E1의 재학습 특징 행렬 = E2의 재학습 특징 행렬이다. 따라서 두 팔의 바깥 Fold 성능 차이는 **트리 수(`n_trees`) 차이**에서만 생긴다.
- **E2는 라벨 혼입 제거 한 가지만 바꾼 것이 아니다.** E1 대 E2에서 ES 단계가 다른 점은 세 가지다. ① tr 특징에 val 라벨이 섞이는지(E1 있음, E2 없음), ② val 행의 특징 생성 방식(E1: 학습 Fold 전체 `KFold` OOF 값, E2: tr 전체 표로 **조회**한 값), ③ 지역 구성·target rate 표를 만드는 기준 행 수(E1 약 166k, E2 약 133k; ES 모델 학습 행은 둘 다 tr 약 133k). 따라서 **E1과 E2의 트리 수·바깥 성능 차이를 라벨 혼입이라는 단일 원인에 귀속하지 않는다.** 이 설계는 "노트북의 ES 프로토콜을 학습 Fold 안에서 tr 기준으로 분리한 판이 트리 수와 바깥 성능을 얼마나 바꾸는가"를 기술하며, 세 요소의 분리는 하지 않는다.
- 또한 E2의 ES 단계 특징(tr 기준 표)은 최종 재학습 특징(학습 Fold 전체 표)과 표 기준 행 수가 달라, E2의 `n_trees`가 최종 학습에 최적이라는 보장은 없다. 이는 E2의 한계로 기록한다.
- 하이퍼파라미터 8개 조합 선택에 대한 라벨 혼입 영향은 이 설계가 다루지 않는다(재탐색 금지). 따라서 **E2가 "혼입 없이 처음부터 고른 설정"의 성능을 뜻하지 않는다**(§13).
- 보조 측정: 두 팔의 **내부 검증 Macro F1**(E1: 라벨 혼입이 있는 val, E2: 혼입 없는 val)과 같은 Fold의 바깥 Macro F1을 나란히 보고하여 낙관 크기를 기술한다. 특징 수준 탐침(§7)에서 "val 라벨을 바꿨을 때 tr 특징이 바뀌는 비율"을 학습 없이 측정한다.
- 같은 `n_trees`가 나온 Fold에서는 `deterministic=true` 설정상 두 팔의 최종 확률이 비트 단위로 같아야 한다. 같지 않으면 감사 신호로 쓴다(§10).

### 3.5 포팅 규칙과 의도된 차이 (전부 기록)

우리 실행기는 노트북 셀 25의 논리를 **옮겨 구현**한다. 의도된 차이:

1. 데이터 읽기: 노트북은 원본 CSV를 직접 읽고 전체 80/20로 분할한다. 우리는 `load_development()`와 저장된 Fold만 쓴다. 우리 개발 자료에는 홀드아웃 행이 없다.
2. 분할 순서·KFold 구성: 학습 Fold는 `building_id` 순으로 정렬돼 있다. 노트북의 `KFold`/`train_test_split`은 위치 기반이라 **같은 시드여도 구성원이 노트북 실행과 다르다**. 시드(42)와 인자는 그대로 둔다.
3. 학습 크기: 노트북 학습 208,480행(내부 tr 166,784)에 대해 우리 학습 Fold는 166,784행(E1·E2의 early stopping tr 133,427, val 33,357)이다. 이 차이는 의도하지 않은 것이 아니라 공통 Fold 비교를 위한 것이며 §9에서 해석에 쓴다.
4. 라이브러리: 노트북은 더 새로운 LightGBM(및 Python 3.14.6, `c:\earthquake\.venv`)을 썼다. 우리는 동결 환경(Python 3.13.5, `lightgbm 4.6.0`, `scikit-learn 1.8.0`, `pandas 2.2.3`, `numpy 2.3.5`, `joblib 1.5.3`)을 쓴다. 수치 완전 일치는 목표가 아니다.
5. 스레드: 노트북 `n_jobs=-1`, 우리는 4(§5).
6. `deterministic=true`, `force_col_wise=true`를 추가한다(실행 간 재현성). 모델 품질을 의도적으로 바꾸려는 것이 아니다.
7. 예측: `predict` 대신 `predict_proba` argmax를 쓴다(확률 저장).
8. 구현 최적화는 **원본 함수 출력과 비트 동일**일 때만 허용한다(§7 대조 확인). `groupby(...).agg(lambda ...)`가 느리면 벡터화를 쓸 수 있으나, 대조 확인이 한 자리라도 어긋나면 원 논리 그대로 쓴다.
9. 잔여 구조적 성질은 **그대로 재현**한다(수정하지 않음): ① target rate OOF 단계의 전역 대체 비율(`overall_high/low`)이 학습 전체(해당 행 포함) 평균이다. 라벨은 학습 행 안에서만 쓰이므로 바깥 검증 행에는 영향이 없다. ② OOF 표의 최소 개수 기준은 학습의 4/5에서, 검증 행 표는 학습 전체에서 센다(개수 규모가 25% 다름). ③ age 995를 포함한 원시 `age`가 `weak_age`와 `*_age_vs_region`에 그대로 들어간다. 이 세 가지는 팀원 설계의 특성으로 기록만 하고 판단하지 않는다.
10. 노트북의 XGBoost·CatBoost·랜덤포레스트·로지스틱 회귀 팔과 5모델 비교, 셀 16·19의 별도 실험은 재현 범위 밖이다. LightGBM 최고 설정만 다룬다.

## 4. 입력과 비교군 (재학습 없음)

| 입력 | 값 |
|---|---|
| 노트북 | `earthquake/data/지진.ipynb`, SHA-256 `480d63cd6adc3f68f70142ff159d229654db5d17a27d3af093baa38b831ac07b`(작성 중 다시 계산해 일치 확인). `prepare`가 `source_snapshot/`에 복사하고 SHA를 검증한다. 셀 25 소스는 별도 텍스트로 추출하고 그 SHA도 봉인한다. 사본은 비공개이며 Git에 올리지 않는다. |
| 환경 | 동결 Python `local_reproduction/20261014_01/venv`(3.13.5). 위 §3.5 버전. |

비교군은 모두 같은 `split_manifest.csv` Fold이며 `prepare`가 경로·SHA·행 정렬(`building_id`, `cv_fold`)·정답 라벨 일치를 확인한다.

| 비교군 | CV Macro F1 | Recall3 | 3→1 | OOF 위치(`local_reproduction/` 기준)와 SHA-256 |
|---|---:|---:|---:|---|
| T8 LGBM-N | 0.699342 | 0.647648 | 530 | `T8_20261014_native_cat/run/stratified/N/oof.csv.gz`, `263680540b07a5cc4fe84dd294ff90ae0da7c9c7b2886f0df733edf9465f9558`(파일 SHA 직접 계산, T8 `analysis.json`과 일치) |
| T14 RF+TE+ROS | 0.700427 | 0.673016 | 744 | `T14_20261015_te_ros_rf_repro/run_v2/strat/RF_TE_ROS/oof.csv.gz`, `69a74d410da08bf06c9da35211e3fe6f1e424bd4afbc72c4de2d2d3c21c09969`(파일 SHA 직접 계산, T14 `run_v2/analysis.json`과 일치) |
| LR-C(연구 기준선) | 0.688687 | 0.639794 | 452 | `20261014_01/private_original/age995_abc_v1/C/lr/oof_predictions.csv.gz`, `de43f9974e536b3a9c5ebac35b984ed886ba2d5426d59dda989ca12f366a2cbc`(파일 SHA 직접 계산, `age995_verification.json`과 일치) |
| T0 LR C=1 (= T1 A 재현 OOF) | 0.683677 | 0.638433 | 467 | `20261014_01/private_original/age995_abc_v1/A/lr/oof_predictions.csv.gz`, `e8dbaa962fe4b2e2fdad160d25f72cbfd0816d73aafbc0419149c53b8a3de221`(파일 SHA 직접 계산, `age995_verification.json`과 일치). 원래 T0의 OOF는 미확보이며 T1 A가 같은 설정의 새 OOF다. |
| (참고) LGBM-O 원핫 | 0.669847 | 0.590077 | 303 | `20261014_01/private_t4_original/private/model_families_v1/LGBM_C_v1/oof_predictions.csv.gz`, `0bc63eb5122afb479c41826918a9fbb43a3475351a934d847833656eec6930cf`(파일 SHA 직접 계산) |

수치 근거는 `reports/STRATIFIED_SUMMARY_20261016.md`(표2·표1)이다. 비교군 숫자는 `prepare`/`finalize`가 OOF에서 다시 계산해 위 값과 일치함을 확인한다. 이 비교는 **설계가 여러 요소에서 다른 모델 간의 기술 비교**다(전처리·모델·범주 처리·지역 변수·가중치가 한 요소가 아니라 여러 개 다르다). 통제된 단일 요인 비교가 아니다.

## 5. 학습 횟수·시도·시간·병렬

- **신규 fit 20회**: E1·E2 각 Fold 5 × (early stopping fit 1 + 재학습 fit 1) = 팔당 10, 합계 20. 특징 구성 A는 Fold당 1회(두 팔 공유), 특징 구성 B는 Fold당 1회. 학습 없는 특징 계산이므로 fit이 아니다.
- **시도 상한 24 fit**: 기본 20 + 수동 복구 최대 2 unit(unit = 팔×Fold = 2 fit, 4 fit). **자동 재시도 0회.** 실패하면 중단하고 보고한다. 비교군 재학습 0. 탐색 추가 fit 0.
- **예산 밖 항목**: 사전 점검의 합성 400행 프로브 fit(범주 파라미터 실효값 확인용)과 합성 데이터 단위 테스트 학습은 연구 학습이 아니며 위 20/24 fit 예산에 **포함되지 않는다**. 별도 기록(`preflight/`)으로 남기고 결과 판정에 쓰지 않는다.
- **전체 CV**: 모든 수치는 개발 208,480행 전체의 5-Fold 교차검증(Fold당 학습 166,784 / 검증 41,696)이다. 표본 CV(일부 행만 쓰는 점검)는 이 연구 결과에 쓰지 않으며, 표본 점검을 하더라도 연구 값과 섞어 보고하지 않는다(AGENTS).
- **병렬은 한 단계만**: LightGBM `n_jobs=4`(제안, 승인 항목). 특징 계산·평가·저장은 단일 스레드, Fold와 팔은 순차 실행, 외부 병렬 없음. 희소 행렬을 쓰지 않는다(밀집 DataFrame 약 166,784 × 75). `.toarray()` 없음.
- **예상 시간(미측정)**: 근거 기록은 두 가지다. ① T8 LightGBM 300 rounds(`n_jobs=2`, 약 16.7만 행 × 39열) fit 중앙값 10.1초, 최대 12.6초(`attempts.jsonl` 25회). ② T18 LightGBM fit 단위 약 13~14초(저장 포함). 이 설계는 learning rate 0.03, 최대 2000 rounds, 75열(colsample 0.5)이라 rounds 수는 알 수 없다(§2 트리 수). 노트북 자체 보고는 8조합 + 재학습에 약 2.7분(하드웨어 미상)이다. 가정: early stopping fit 수십 초, 재학습 수십 초, **특징 구성 계산이 Fold당 두 번**이며 pandas `groupby(...).agg(lambda ...)`(범주 수천~수만 개, 키 9개 × 표 6개)가 병목일 가능성이 있다. 이를 종합한 **거친 추정은 Fold당 약 5~20분, 전체 약 0.5~2시간**이며 **측정하지 않았다**. 첫 Fold 실측으로 추정을 고쳐 보고하고, 추정이 틀려도 설계·판정은 바뀌지 않는다.
- 디스크: LightGBM 텍스트 모델 20개와 특징 행렬, 확률 파일 정도라 작다(수 GB 미만). `prepare`가 여유 공간 ≥ 20 GB를 요구한다(제안).

## 6. 체크포인트·원장·재개

- **unit = (팔, Fold).** 단계 체크포인트는 `es_done`(early stopping 결과 저장), `complete`(재학습·확률 저장) 두 번이다. 중단 후 재개는 완료 단계를 해시 검증만 하고 다시 학습하지 않는다(`es_done` unit은 재학습 fit만 수행. 이때도 early stopping fit 횟수는 늘지 않는다).
- Fold/팔 완료 즉시 저장하는 산출물:
  - `{arm}_fold_{f}_es_model.txt`, `{arm}_fold_{f}_es.json`(best_iteration, 내부 검증 Macro F1, `multi_logloss` 곡선 요약, tr/val `building_id` 목록의 SHA, `cap_hit`, 소요 초).
  - `{arm}_fold_{f}_final_model.txt`, `{arm}_fold_{f}_pred.csv.gz`(바깥 검증 `building_id`, `cv_fold`, 정답, 확률 3개, 예측; 17자리 round-trip), `{arm}_fold_{f}.json`(지표·혼동행렬·파라미터·진단·소요 시간·모델 텍스트 실효 파라미터).
  - `fold_{f}_features_state.joblib`(특징 구성 A의 지역 구성 표, target rate 표, 범주 어휘, 열 이름·순서)와 바깥 검증 특징 행렬의 해시(및 감사용 행렬 사본). 특징 구성 B도 같은 형식으로 팔 E2의 early stopping 입력으로 보존한다.
- 모든 산출물의 SHA-256을 `ledger.json`·`attempts.jsonl`에 기록하고 `backup/`에 복사한다. 원본 실행 결과는 덮어쓰지 않는다. 저장 직후 **재로딩 검증**: 모델 텍스트를 다시 읽어 바깥 검증 행에 대해 확률이 저장 값과 `atol=1e-12` 이내, 예측 등급 일치, CSV `float_precision='round_trip'` 재읽기 정확 일치. 추가로 **트리 수 교차 확인**: 재학습 모델의 반복 수가 `n_trees`와 같고(클래스당 트리 `n_trees`개, 총 `3 × n_trees`개), `predict(num_iteration=n_trees)`가 기본 `predict()`와 일치한다. ES 모델은 `best_iteration` 이후 patience만큼 더 학습한 트리를 포함하므로, `predict(num_iteration=best_iteration)`가 기본 예측과 같고 기록된 val `multi_logloss`·내부 Macro F1과 일치함을 확인한다. 이 접두 관계는 같은 모델 안에서의 확인이며, ES 모델과 재학습 모델은 학습 행·특징이 달라 서로의 접두가 아니다.
- 완료 Fold는 재학습하지 않는다. `RUNNING.lock`으로 중복 실행을 막고, `running` 상태의 미완 산출물은 `.incomplete_attempt<n>`으로 치운 뒤 수동 재개한다. 학습 예산(20 fit, 상한 24)·미등록 unit 실행은 코드가 거부한다.
- 설계 봉인(`design.json`/`design_seal.json`): 설정, 소스 SHA(노트북 사본 SHA, 셀 25 추출본 SHA, 우리 실행기·설정·이 문서 SHA), 환경, 분할 SHA, 비교군 SHA, 특징 열 순서(§3.1), 판정 임계, 사전 점검 결과를 학습 전에 고정한다. 봉인 뒤 코드·설정·계획은 바꾸지 않는다.

## 7. 사전 점검 (학습 없음, `prepare` 전 PASS 필수)

1. **원본 대조(비공개 사본 필요)**: 사본 노트북에서 셀 25의 함수 정의(`add_building_features`, `add_one_level`, `add_region_composition`, `damage_rate_table`, `add_target_rates`, `make_weights`)만 AST로 추출해 합성 데이터와 **실데이터 Fold 0**에서 우리 포팅 출력과 비교한다. 함수만으로는 실행되지 않으므로 추출 대상에 **모듈 상수와 import**를 함께 명시한다: 상수 `SEED = 42`, `MIN_BUILDINGS = 20`, `MIN_COMBO = 50`(`damage_rate_table`의 기본 인자가 `MIN_COMBO`를 참조), import `pandas as pd`, `KFold`(`sklearn.model_selection`). 상수·import 목록과 추출 텍스트의 SHA를 봉인에 기록하며, 이 외의 전역(그래프·모델 라이브러리 import 등)은 추출하지 않는다. 75열 이름·순서·값이 정확히 같아야 한다(오차 허용 0, 부동소수는 비트 동일 또는 `atol=0`). 이 대조는 비공개 사본이 있을 때만 가능하므로 공개 단위 테스트에는 넣지 않는다. 또한 이 대조는 **동결 환경(pandas 2.2.3 등)에서 실행한 두 구현의 일치만 보증**하며, 노트북 환경(pandas 3 계열 추정, `str` dtype)에서의 동작 일치까지는 보증하지 않는다. 단위 테스트는 사본 없이 합성 데이터의 수작업 기대값으로 구성하고, 사본이 필요한 항목은 skip으로 분리해 pass/skip을 구분한다(AGENTS).
2. **입력 점검**: 학습·검증 ID 서로소, 학습 Fold 166,784 / 검증 41,696, 지역 ID 정수 왕복(`str(int(x)) == x`), 이진 열 값 0/1, 홀드아웃 경로 가드(`final_holdout` 읽기 시 즉시 실패).
3. **행렬 점검**: 특징 구성 A·B의 열 수 75, 열 이름·순서가 §3.1 목록과 일치, 결측 NaN은 범주 어휘에 없는 값 외에는 0개, 범주 10열 어휘 크기, 학습 Fold 지역 구성·target rate 표의 크기와 작은 조합 대체 개수.
4. **라벨 혼입 탐침(학습 없음)**: Fold 0에서 val 행의 라벨을 임의로 섞은 판을 만든다.
   - 특징 구성 A 생성 순서(E1)에서 tr 행 특징이 val 라벨 변화에 따라 바뀌는 행·열 비율을 기록한다(혼입 존재 확인과 크기 기술).
   - 특징 구성 B 생성 순서(E2)에서 tr·val 특징이 val 라벨 변화에도 **변하지 않음**을 확인한다.
   - 특징 구성 A에서 바깥 검증 행 특징이 바깥 검증 라벨 변화에도 변하지 않음을 확인한다.
5. **환경 점검**: 동결 환경 버전과 `lightgbm` 4.6.0의 `fit(eval_set=...)`·`best_iteration_` 동작, 합성 400행 프로브로 범주 파라미터 실효값을 읽어 설계에 기록(프로브 fit은 20/24 fit 예산 밖, §5).

점검이 하나라도 실패하면 `prepare`와 학습을 하지 않고 중단·보고한다.

## 8. 지표와 사전 고정 기술 판정

### 8.1 보고 지표 (각 팔, Fold별·평균·풀링)

Fold별 Macro F1(labels [1,2,3]), 평균, SD(ddof=1), 풀링 Macro F1, Accuracy, 클래스별 Precision/Recall/F1/지원 수, 혼동행렬, Recall3, 3→1·3→2 건수와 비율(분모 = 실제 3등급 수 69,774), 예측 비율, 같은 행 turnover(`experiments/geo_ablation/audit.py`의 `turnover()`)를 비교군 각각과 계산한다. 이어서 §3.4 보조 측정: 팔별 `n_trees`, 내부 검증 Macro F1, 내부-바깥 차이, 라벨 혼입 탐침 비율. age 995 집단 지표와 geo1 지역 26/20/13 프로파일은 기존 보고와 같은 형식으로 기술한다(원시 `age` 995가 파생 변수로 전달되는 파이프라인이라 관찰 항목으로 둔다).

### 8.2 사전 고정 기술 판정 (임계는 제안값, 승인 항목)

모든 비교는 **같은 Fold의 Fold 단위 ΔMacro F1 = (T33 팔) − (비교군)**이다. 유의성·인과 주장을 하지 않으며, 이 라벨은 채택·권고가 아니다.

- **D1 비교군별 기술 라벨**(각 팔 × 각 비교군 T8-N·T14·LR-C·T0): 평균 Δ ≥ +0.005 **그리고** 5 Fold 중 4 Fold 이상 팔이 엄격히 큼 → `ADVANTAGE_DESCRIPTIVE`. 평균 Δ ≤ −0.005 **그리고** 4 Fold 이상 팔이 엄격히 작음 → `DISADVANTAGE_DESCRIPTIVE`. 그 외 `INDISTINGUISHABLE`. (T8 §7 규칙을 차용한 제안값.)
- **D2 안전 지표는 판정 없이 병기**: 풀링 Recall3, 3→1·3→2 건수와 비율을 비교군과 나란히 보고한다. 3→1 허용 상한은 오류 비용 판단이라 이 실험에서 정하지 않는다(T29 사용자 결정 항목). 가중치 0.3이 3→1을 늘릴 가능성이 있다는 점은 **결과가 아니라 예상**이며, 증가 여부는 관찰로만 기록한다.
- **D3 ES 프로토콜 차이의 영향**(E1 vs E2): 평균 Δ(E1−E2)의 절댓값 ≤ 0.001이고 Fold별 |Δ| 최댓값 ≤ 0.003이면 `ES_PROTOCOL_EFFECT_SMALL`, 아니면 `ES_PROTOCOL_EFFECT_NOTICEABLE`. 트리 수 차이는 비율로 병기한다. 이 라벨은 라벨 혼입·val 특징 생성 방식·지역 표 기준 행 수의 **합한 효과**를 기술하며 단일 원인에 귀속하지 않는다(§3.4).
- **D4 노트북 단일 분할 값과의 거리(참고 라벨, 판정 아님)**: E1 CV 평균과 0.7143의 차이 `|Δ|` ≤ 0.005 `NEAR`, ≤ 0.015 `MODERATE`, 그 밖 `FAR`. 이 거리는 §9 원칙대로 분할 차이와 선택 편향으로만 설명하며 "재현 성공/실패"로 부르지 않는다. **출처 한계**: 0.7143이 적힌 셀 25의 `execution_count`는 `null`이고 출력이 여러 세션에서 섞여 있어, 이 값이 현재 셀 소스를 위에서 아래로 실행한 출력인지 확인할 수 없다. 따라서 0.7143은 "노트북에 저장된 보고값"으로만 쓴다. 같은 노트북의 두 번째 참고값으로 **셀 23의 가중치 없는 XGBoost 기준 0.7028**(75개 변수, 500 rounds, lr 0.1, depth 6, 같은 노트북 분할; 셀 23은 `execution_count` 14이며 셀 25 출력의 기준선 표기 0.7028과 일치)을 D4와 함께 제시한다. 보고서는 두 값과 E1·E2 CV 평균을 나란히 적는다.
- **주 보고 팔**: 비교군 대비 라벨은 두 팔 모두 보고한다. 보고서는 E1(충실 재현)과 E2(ES 분리판)를 동등하게 병기하며, 비교군 대비 대표 줄은 E2(내부 검증에 라벨 혼입이 없는 ES 프로토콜)로 두되 E1과의 차이를 단일 원인으로 설명하지 않는다(승인 항목). 두 팔의 바깥 점수는 모두 바깥 검증 행 라벨을 쓰지 않는다(§3.4).
- 라벨 충족 여부와 무관하게 **모든 수치를 보고**한다. 홀드아웃 평가, 최종 모델 교체, 표준 5조건 채택 판정은 하지 않는다.

## 9. 노트북 단일 분할 값(0.7143)과의 차이를 해석하는 원칙

노트북 값과 우리 CV 평균은 **같은 양이 아니다**. 차이를 설명할 때 아래 요인만 나열하고, 어느 요인이 얼마나 기여했는지는 이 설계로 분리되지 않는다고 적는다.

1. **분할·행 구성**: 노트북의 테스트 52,121행은 전체 260,601행에서 무작위로 뽑은 한 번의 분할이다. 우리 검증은 개발 208,480행의 5-Fold이고 홀드아웃 52,121행은 쓰지 않는다. 두 집합의 구성원이 같다는 근거는 없다(확인하지 않음, 홀드아웃 접근 금지).
2. **학습 크기**: 노트북 최종 학습 208,480행 대 우리 학습 Fold 166,784행.
3. **단일 분할 대 5-Fold 평균**: 단일 분할 값은 한 번의 표본 변동을 포함한다(우리 기존 후보의 Fold SD는 약 0.001~0.003).
4. **선택 편향**: 노트북은 5개 모델 중 테스트 점수 최고를 골랐고(1~2위 차 0.0012) 설정 8개를 검증 점수로 골랐다. 이 검증 점수는 §3.4의 라벨 혼입으로 낙관적일 수 있다.
5. **정보 유입**: 노트북의 설정(가중치 0.3, 위 LightGBM 설정, 변수 구성)은 **전체 260,601행의 무작위 80/20 분할**에서 골랐다. 그 학습·검증·테스트 행은 우리 개발 Fold의 검증 행뿐 아니라 **우리 과거 홀드아웃 행과도 겹쳤을 가능성이 높다**(구성원 일치는 확인하지 않음, 홀드아웃 접근 금지). 따라서 선택된 설정에는 우리 Fold 검증 라벨과 홀드아웃 라벨의 영향이 일부 섞였을 수 있다. 결론: **이 벤치마크의 점수도, 이 파이프라인을 후보로 올려 이후 (홀드아웃 또는 새로 설계하는) 최종 평가를 하더라도 그 결과도 독립 성능이 아니다.** 우리 Fold 점수는 선택 통계다.
6. **환경**: 더 새로운 LightGBM·Python·pandas, 스레드 수, KFold 구성원 차이(§3.5).
7. **반복 사용**: 우리 Fold는 이미 수십 개 설정의 비교에 사용됐다(`STRATIFIED_SUMMARY` 한계 1). T33 점수도 독립 성능이 아니다.

해석 문장 형식(보고서 규칙): "이 점수는 노트북 단일 분할 값의 재현 여부를 판정하지 않으며, 같은 파이프라인을 우리 공통 Fold에서 계산한 값이다. 차이는 위 요인으로 설명될 수 있으나 분리하지 못했다."

## 10. 독립 감사 (공개 보고 전 필수)

감사는 학습 담당과 다른 주체가 수행하고 새 fit·joblib 역직렬화·실행기 import 없이 진행한다(기존 감사 방식).

1. 산출물 SHA·원장·`attempts.jsonl` 대조, 학습 횟수 ≤ 상한, 완료 unit 재학습 0, 홀드아웃 접근 0.
2. 학습 Fold와 검증 Fold의 `building_id` 서로소와 union = 개발 전체, tr/val 서로소와 union = 학습 Fold, E1·E2의 tr/val 구성 동일.
3. 모델 텍스트로 확률 재추론(최대 확률 차이 ≤ 1e-12, 라벨 일치). 같은 `n_trees`인 Fold에서 두 팔 확률 비트 일치.
4. 독립 구현(별도 코드, 벡터화/다른 방식)으로 바깥 검증 행의 특징 75열을 학습 Fold 표에서 다시 계산해 저장된 행렬과 대조. 원시 라벨 교란 탐침의 재실행.
5. OOF에서 지표(Macro F1, 클래스별 P/R/F1, 혼동행렬, 3→1)를 독립 계산, 비교군 값 재계산, 판정 라벨 재도출.
6. 공개 보고 초안이 팀원 코드 원문·행별 예측을 싣지 않았는지 검토.

## 11. 공개 보고 범위

- 공개(독립 감사 PASS 뒤에만): 집계 지표, Fold별 Macro F1과 `n_trees`, 판정 라벨, 한계, 파이프라인을 **요약**한 설명(재료·유형 파생, 지역 구성, 조합별 피해 비율의 학습 Fold 내 OOF 계산, 지역 대비 gap, 변수 75개라는 구성 수준).
- **비공개 유지**: 노트북 원문과 셀 소스, 추출본, 행별 예측·OOF, 모델·특징 행렬·manifest.
- 우리 실행기(`experiments/ey_pipeline_benchmark/`)는 팀원 논리를 옮긴 코드이므로 파일 머리에 **출처와 노트북 SHA를 표기**하되 노트북 주석·변수 설명 문장을 그대로 복사하지 않는다. 실행기를 공개 Git에 올릴지는 사용자 승인 항목이다(§15).
- 표기 원칙: "평가"가 아니라 "재현 벤치마크". 팀원 작업물을 우리 결과에 비해 낮다/높다고 단정하지 않는다. 노트북 수정 없음, 팀원 브랜치 접근 없음.

## 12. 홀드아웃·경계

- 홀드아웃(`earthquake/data/final_holdout/`)은 어떤 단계에서도 읽지 않는다. 실행기는 경로 가드로 이를 막는다. 원시 전체 CSV(`nepal_building_damage.csv`, 260,601행)는 홀드아웃 행을 포함하므로 **쓰지 않으며**, 노트북의 단일 분할 값은 같은 분할로 재현하지 않는다.
- 과거 홀드아웃은 후속 선택·튜닝·early stopping에 재사용하지 않고 이 실험의 독립 성능으로도 쓰지 않는다. 별도 최종 평가 설계(T29 초안)는 사용자 결정 사항이다.
- **독립성 한계**: 노트북 설정은 전체 데이터의 무작위 80/20에서 선택돼 우리 홀드아웃 행과도 겹쳤을 가능성이 높다(§9.5). 그래서 T33 결과는 독립 성능이 아니며, 이 파이프라인을 이후 최종 평가에 올리더라도 그 평가는 "설정 선택 단계에서 이미 정보가 유입된 설정의 평가"로 표기해야 하고 독립 성능으로 주장하지 않는다.
- 기존 브랜치·기존 비공개 결과 변경 금지. 원시 데이터·모델·OOF·manifest·행별 예측의 Git 업로드 금지. push는 메인이 독립 감사 뒤에만 한다. 막히면 설계를 바꾸지 않고 중단·보고한다.

## 13. 한계

1. 재현이 **환경 불일치**(Python 3.14.6·새 LightGBM·pandas 3 추정 대 3.13.5·lightgbm 4.6.0)에서 이뤄지며, 노트북의 정확한 라이브러리 버전은 확인하지 못했다.
2. 노트북의 `best_iteration_`이 기록되지 않아 같은 트리 수로 고정하는 대조는 불가능하다. 트리 수는 우리 Fold마다 새로 정한다.
3. E1·E2의 바깥 성능 차이는 트리 수 차이에서만 생기지만, 그 트리 수 차이는 ES 프로토콜 세 요소(라벨 혼입, val 특징 생성 방식, 지역 표 기준 행 수)의 합이라 단일 원인에 귀속하지 못한다. **하이퍼파라미터 선택 단계의 라벨 혼입 영향은 측정하지 않는다**(재탐색 금지).
4. 지역 ID를 수치로 넣는 구성이 가이드의 "지역 코드의 크기를 거리·순위로 해석하지 않는다"는 주의와 충돌하지 않는지는 **해석** 문제다. 트리 모델이 순서 분할을 쓰는 것은 사실이며, 재현을 위해 그대로 두고 크기에 의미가 있다는 해석을 하지 않는다. 지역 구성·target rate 변수는 학습 Fold의 지역 분포에 크게 의존한다.
5. 층화 분할은 같은 지역이 학습·검증에 모두 나온다. 이 파이프라인의 지역별 target rate는 **처음 보는 지역**에서 학습 시와 다른 분포를 낼 수 있다(T10·T14에서 관찰한 현상). geo2/geo1 일반화는 이 실험에서 평가하지 않으며 별도 사용자 결정이다.
6. 같은 5-Fold를 다른 후보들과 반복 사용한 선택 통계이며 독립 성능이 아니다. Fold 5개는 독립 표본이 아니라서 승수·SD를 유의성으로 읽지 않는다.
7. 3→1 같은 중대 오류는 Macro F1에 반영되지 않는다. 기술 라벨이 이를 가리지 않도록 병기한다.
8. 지진 강도·지반 조건은 반영되지 않았고 인과·안전 판정이 아니다.
9. 후속 분해 실험(예: EY 변수 75개만 고정하고 T8 하이퍼파라미터 사용, 모델 계열별 EY 변수 효과)은 이 실험의 범위 밖이며 별도 사전등록이 필요하다. 노트북 변수 집합은 범주 처리·지역 ID 표현·age 995 처리·가중치·튜닝이 동시에 다르므로 **한 요소 비교가 아니다**.

## 14. 실행 순서 (승인 후, 명령은 구현 때 확정)

1. 단위 테스트(합성 데이터) → `python -m unittest discover -s tests -v`에서 pass/skip 구분 보고 → 계획·코드 커밋.
2. `source_snapshot/` 생성(노트북 사본 SHA 검증, 셀 25 추출본).
3. 사전 점검(§7) PASS.
4. `prepare`: 전체 커밋 SHA로 설계 봉인.
5. `fit-all`: Fold 0부터 순차. 첫 Fold 실측 시간 보고.
6. `finalize`: 팔별 OOF·지표·판정 라벨이 담긴 분석 JSON.
7. 독립 감사(§10) → PASS 뒤에만 보고서·집계 JSON 커밋. 푸시는 메인이 조율한다.

## 15. 승인이 필요한 항목

1. 실험 수행 여부와 범위: 필수 팔 E1(충실 재현)·E2(ES 분리판)만(기본), 신규 fit 20회, 시도 상한 24 fit(합성 400행 프로브·단위 테스트 학습은 예산 밖).
2. 팔 정의 확정: E2의 tr/val을 학습 Fold의 원시 행에서 먼저 나누고 ES 단계 특징을 tr 안에서만 계산한다는 정의(val은 조회, 지역 표는 tr 기준), 팔 이름 E2 "ES 분리판"과 D3 라벨 `ES_PROTOCOL_EFFECT_*`, 재학습 특징은 두 팔 모두 학습 Fold 전체(특징 구성 A)로 한다는 정의.
3. 하이퍼파라미터 고정: 노트북 LightGBM 최고 설정(weight 0.3, lr 0.03, leaves 31, min_child 20, subsample 1.0, colsample 0.5, `subsample_freq 1`, seed 42), 최대 2000 rounds, patience 50. 트리 수를 우리 Fold의 early stopping으로 정하는 것(노트북 값이 없으므로 기본).
4. 의도된 차이 승인: 지역 ID를 정수 수치로 복원해 입력, 이진 열 int64 복원, `predict_proba` argmax, `deterministic=true`·`force_col_wise=true` 추가, 벡터화 최적화는 원본과 비트 동일일 때만 허용.
5. LightGBM `n_jobs=4`와 Fold·팔 순차 실행(한 단계 병렬).
6. 사전 점검에서 원본 함수 AST 추출본 실행 허용(팀원 노트북 사본의 함수 정의만 비공개 폴더에서 실행, 데이터는 학습 Fold만).
7. 노트북 사본을 `local_reproduction/T33_20261016_ey_pipeline/source_snapshot/`에 보존하는 것, 공개 Git에 올리지 않는 것.
8. 우리 실행기의 공개 Git 반영 여부(출처·SHA 표기 방식), 팀원 EY에게 사용·재구현 사실을 알릴지 여부. 저장소 권한·공개 범위·팀원 배정은 바꾸지 않는다.
9. 판정 임계: D1(±0.005, 4/5 Fold), D3(0.001, 0.003), D4(0.005, 0.015). 3→1 상한은 정하지 않는다.
10. 보고 팔 표기(E1·E2 병기, 대표 줄 E2, 단일 원인 귀속 금지), 보고서 표기 문구("EY 노트북 파이프라인 층화 CV 재현 벤치마크").
11. 예상 시간·디스크 여유 기준(미측정, 첫 Fold 실측 후 갱신)과 다른 학습 실험과 동시 실행하지 않는다는 운영 원칙.
12. 결과가 좋아도 T29 확인 평가 후보에 이 파이프라인을 추가할지, 공식 개인 최종 모델(T0 LR C=1)을 유지할지는 별도 사용자 결정으로 둔다. 이 실험은 둘 중 어느 쪽도 바꾸지 않는다.
13. 후속 분해 실험(EY 변수 집합의 효과를 하이퍼파라미터·범주 표현과 분리) 필요 여부는 결과를 본 뒤 별도 사전등록으로 결정한다.

## 16. 이 문서 작성 시 확인하지 못한 사항

- 노트북 실행 환경의 LightGBM·pandas·numpy 정확한 버전, 하드웨어, 셀 25의 출력이 어느 커널 세션에서 나왔는지.
- 최고 설정의 `best_iteration_`, 시험 8개 조합 각각의 트리 수.
- EY 테스트 52,121행과 우리 홀드아웃 52,121행의 구성원 일치 여부(홀드아웃 접근 금지로 확인하지 않음).
- 이 파이프라인의 실제 학습·특징 계산 소요 시간(미측정).
- `peter` 현재 HEAD(작성 중 git을 조회하지 않음). 봉인 시 전체 40자 커밋 SHA로 결속한다.
