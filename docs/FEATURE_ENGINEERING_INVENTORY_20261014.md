# Earthquake 개인 실험 파생변수 인벤토리 (코드 근거)

확인일: 2026-10-14 KST. 이 문서는 `peter`의 원본 38개 입력 및 T1~T6 개인 연구의 **실제 모델 피처/전처리 변환/검증 그룹**을 구분한다. 팀 공통 채택 여부를 뜻하지 않는다. 연구 중복 방지를 위한 로컬 에이전트 참고용.

## 0. 먼저 구분할 세 가지

- **원본 피처**: `preprocess.py`의 38개 입력. 건물 ID 및 정답은 제외. 범주형 원핫 인코딩으로 만들어진 약 12,5xx개 열은 **원본 범주를 인코딩한 열**이지, 수천 개의 독립적인 수작업 파생변수라는 뜻이 아니다.
- **새 모델 입력으로 실제 실험한 파생변수**: 아래 `age_is_995` 및 `age_band`, `geo_foundation`, `geo_age`, `foundation_age` 5종. G1~G5는 이 4종 범주 피처의 사용 조합이며 5종을 동시에 적용한 경우는 G5다.
- **전처리 값 변경·EDA·그룹 분할**: 중앙값 대체, 연령 상한절단, 지역 그룹키 생성, 피해 비율 집계 등은 각각 모델 입력 열의 추가와 다르다. 이 차이를 숨기지 않는다.

## 1. 실제 연구에서 만든 모델 입력

| 이름 | 정의 / 코드 | 도입 실험 | 적용된 경우 | 주의 |
|---|---|---|---|---|
| `age_is_995` | `(age == 995).astype(float32)`; 995 여부 0/1 | T1 B·C / T1 E1~E3 등 flag 사용안 | B: 995 원값 유지 + 표시; **C: 비995 학습 Fold 연령 중앙값으로 995만 대체 + 표시** | 995를 공식 결측 코드라고 확정하지 않음; 개인 최초 최종 모델 `peter`에는 미적용 |
| `age_band` | **대체 전 원본** `age`를 `0,1–10,11–20,21–30,31–50,51–100,101–994,995`의 8개 범주로 나눔 | T3 G1~G5 | 원핫 인코딩으로 추가 | `age=995`가 중앙값 15년 구간으로 옮겨가지 않음 |
| `geo_foundation` | `str(geo_level_1_id) + '|' + str(foundation_type)` | T3 G2·G5 | 1차 지역 × 기초구조 복합 범주를 원핫 추가 | `foundation_type`은 실제 재료명으로 복원하지 않은 익명 코드 |
| `geo_age` | `str(geo_level_1_id) + '|' + age_band` | T3 G3·G5 | 1차 지역 × 연령 구간 복합 범주 | 원본 나이가 아닌 8개 구간과 결합 |
| `foundation_age` | `str(foundation_type) + '|' + age_band` | T3 G4·G5 | 기초구조 × 연령 구간 복합 범주 | 지역 변수와 결합한 3중 교차항은 없음 |

정의 구현: [T1 `Age995Treatment`](../experiments/age995/run.py), [T3 `raw_terms` / `Terms`](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/experiments/age995/interactions/run.py), [T3 고정설정](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/experiments/age995/interactions/config.json).

## 2. T3 모델별 입력 조합과 실제 결론

| 설정 | C 전처리 기준 추가 항목 | 5-Fold CV Macro F1 | 판정 |
|---|---|---:|---|
| C | `age_is_995` + 995 학습 Fold 중앙값 대체, 원본 38개 | 0.688687 | **후속 연구 기준 유지** |
| G1 | C + `age_band` | 기존 C보다 평균 개선 | 탐색적 비교 |
| G2 | C + `age_band` + `geo_foundation` | **0.690993** | 평균 최고이지만 3→1 오류 452→478, 995 집단 F1 악화; 미채택 |
| G3 | C + `age_band` + `geo_age` | 기존 C보다 평균 개선 | 탐색적 비교 |
| G4 | C + `age_band` + `foundation_age` | 기존 C보다 평균 개선 | 탐색적 비교 |
| G5 | C + 위 네 파생 범주 전체 | 기존 C보다 평균 개선 | 탐색적 비교 |

정확한 전 설정 점수와 오류 전이는 [T3 결과](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_INTERACTIONS_20261014.md) 확인. **G2 개선을 `geo_foundation` 단독 효과로 해석하지 말 것:** G2에는 `age_band`도 들어간다. 적절한 직접 대조는 G2와 G1이다.

## 3. 파생 '열'이 아니라 실험적 전처리/검증인 것

| 작업 | 실제 수행 범위 | 모델 입력 추가 여부 |
|---|---|---|
| 기존 원핫 인코딩 | 지역 3개 + 문자 범주 8개; 각 학습 Fold에서 범주 학습, 미관측 범주는 0 | 새 수작업 파생 피처로 계산하지 않음 |
| 학습 Fold 중앙값 대체 | C: 995 행만 비995 연령 중앙값(각 Fold 실측 15)으로 대체. 0은 중앙값 계산에 포함 | `age_is_995`만 추가 |
| 상한 절단 | T1 E 정책: 비995 분포의 상위 분위수에서 상한 계산 후 995만 또는 상한초과 행을 절단, 설정에 따라 표시 포함 | 상한값 변경이 주 효과이며 새 종류의 열은 없음 |
| 행 삭제 | T1 D/F: 995 또는 FAITH 논문 임계 조건으로 훈련 행 삭제, 검증행 보존 | 열 추가가 아니라 훈련 표본 변경 |
| SMOTE/RandomOverSampler | `peter` 후속 실험의 학습용 샘플 추가 및 보간; 최종 미채택 | 원본 추론 스키마의 새 열이 아님 |
| 피해 3등급 비율/age 구간 EDA | `reports/EDA_FINDINGS_20261013.md`의 집계·오류 진단 | 정답 기반 집계를 학습 피처로 사용하지 않음 |
| T5 지역 Group CV 복합그룹 | 분할용 (`geo_level_1_id`, `geo_level_2_id`) 그룹키 또는 1차 지역 단일 그룹 | **분할 키**이며 새 예측 피처가 아님. T3의 `geo_foundation`과 다름 |
| T6 R12/R1/R0 | 범주 인코딩 대상에서 `geo3`, `geo2+geo3`, `geo1+geo2+geo3` 단계 제거 | 파생 추가 아님; **2026-10-14 현재 실제 R12/R1/R0 점수 독립 검증 미완료** |
| T4 XGB/LGBM | 기존 A/C 표현과 동일한 조건의 모델 계열 비교 | 신규 파생 피처 추가 아님 |

## 4. 팀 공유와 로컬 에이전트 주의

- 연구용 `C` / `G2`와 `peter`에서 이미 완료한 **개인 최종 모델(LR C=1, 하이퍼파라미터 C)**은 서로 다른 산출물이다. 개인 최종 모델에 T1~T6의 새 파생변수를 반영했다고 주장하지 않는다.
- `geo_level_1_id` 같은 원본 컬럼, 원핫 더미, `age_is_995` 같은 신설 열을 섞어 '파생변수 1만여 개'라고 집계하지 않는다.
- T1~T3은 개발 데이터의 반복 탐색적 CV다. 팀 공통 최종 모델·Streamlit 앱·`main`을 변경하지 않는다. 52,121행의 이미 사용된 개인 홀드아웃을 재사용하지 않는다.
- 공개 Git에는 원시 CSV·OOF·저장 모델·비공개 ZIP을 올리지 않는다. 수치 비교는 동일 Fold/동일 행·평가 방법에서만 한다.

관련 링크: [T1 정책](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_EXPERIMENT_20261013.md) · [T3 계획](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/experiments/age995/interactions/PLAN.md) · [T5 계획](https://github.com/Peter-jackson12/earthquake/blob/bb0e652947ebf6ab8f8420f819a9f69d81b18265/experiments/geo_group_cv/PLAN.md) · [T6 계획](https://github.com/Peter-jackson12/earthquake/blob/bb0e652947ebf6ab8f8420f819a9f69d81b18265/experiments/geo_ablation/PLAN.md).
