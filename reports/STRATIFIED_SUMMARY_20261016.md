# 층화(Stratified 5-Fold, seed 42) 결과 종합 — 2026-10-16

> **독립 검토 PASS_WITH_NOTES 반영본(2026-10-16).** `reports/`의 감사된 보고서·집계 파일에서 숫자를 옮겨 정리한 문서이며, 신규 학습·재추론은 없다. 독립 검토 지적을 반영했으며 팀 공통 분할·채택 결정을 대신하지 않는다.

실험 ID `A2_20261016_stratified_summary` · 근거 집계: 같은 저장소 `reports/`, `docs/` · 출력 기계가독 판: [stratified_summary_20261016.json](stratified_summary_20261016.json)

## 1. 목적과 평가 기준 변경 사유

- 과제 가이드(`docs/PROJECT_GUIDE_KO.md` §3)는 공통 분할을 **층화 분할**로 두고 필수 네 모델(Logistic Regression, Decision Tree, Random Forest, sklearn GradientBoostingClassifier), 불균형 처리 비교, 주지표 Macro F1을 요구한다. 지역 단위(미관측 지역) 일반화 비교는 심화 항목이다.
- T11 이후의 후반 연구는 지역 일반화를 보는 geo2 Group 분할을 주로 사용했다. 사용자가 **층화 분할을 주 평가 기준**으로 정했으므로(2026-10-16), 층화 seed 42 5-Fold 결과가 있는 모든 설정을 한 곳에 모아 같은 기준으로 비교할 수 있게 한다.
- 이 문서는 새 모델을 학습하지 않는다. 층화 결과가 없는 실험은 표3에 이유와 함께 따로 적는다. geo2 Group 결과는 이 표에 섞지 않는다.

## 2. 공통 조건과 읽는 법

- 자료: 개발 208,480행. 분할: Stratified 5-Fold, seed 42, `split_manifest.csv` SHA-256 `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721`. Fold당 학습 166,784 / 검증 41,696행. 표의 모든 값은 **전체 개발 CV**이며 표본 CV가 아니다.
- 같은 층화 seed 42 Fold인지는 행마다 JSON `split_evidence`에 근거(보고서 기재 SHA·감사 `manifest_sha256`·사전등록 계획)를 남겼다. 다른 seed나 다른 분할(geo2/geo1 Group)의 값은 이 표에 넣지 않았다.
- 주지표 **CV Macro F1** = 5개 Fold F1의 평균(±표준편차, ddof=1, 보고서 규약). Accuracy·Recall3(3등급 재현율)·3→1·3→2 건수는 5개 Fold 검증 예측을 합친(pooled OOF) 값이다. 3→1·3→2의 분모는 실제 3등급 69,774건이다. CV 평균과 pooled F1은 다른 요약이다.
- **Δ(LR-C)** = 해당 행의 CV Macro F1 − LR-C(T1 C, 995만 학습 Fold 중앙값 대체 + 표시, **0.688687**). LR-C 자신과 다른 실험의 비교는 전처리·모델·분할 요소가 한 개가 아니라 여러 개 다를 수 있다. 개별 비교 설계는 각 보고서를 따른다.
- 값은 JSON/CSV 원자료의 부동소수 값을 소수 6자리로 표시했다(반올림 자리: 6). 원자료 전체 자릿수는 JSON에 있다. `—`는 값이 없음이며 이유는 JSON `null_reasons`에 있다.
- **이 표의 순위는 같은 Fold 집합 안에서만 의미가 있다.** 같은 검증 자료로 여러 설정을 반복 비교한 선택 통계이며, 새 독립 성능이 아니다. "최고 성능 확정"이나 채택·권장을 뜻하지 않는다.

## 3. 표1 — 필수 네 모델과 불균형 처리 (T0, 19개 설정)

개인 비교(2026-10-13)의 19개 고유 설정이다. 전처리는 모두 v1.0.1, 모델·처리가 달라 Δ(LR-C) 외에 아래 "한 요소 비교 짝" 표를 함께 본다. 근거 보고서: `reports/BASELINE_REPORT.md`, `reports/FOLLOWUP_REPORT_20261013.md`, `reports/PERSONAL_FINAL_REPORT_20261013.md`, `docs/PERSONAL_MODEL_SELECTION_20261013.md`.

제한 튜닝 5개 행(LR C=0.1/10, RF 3종)의 개별 수치는 보고서 MD에 적혀 있지 않고 `personal_reference_snapshot.json`(및 동일 값의 `dashboard_models.json`) 집계에만 있다.

| 설정 | 모델 계열 | 처리/변경 요소 | CV Macro F1 (±SD) | Accuracy | Recall3 | 3→1 | 3→2 | Δ(LR-C) | 출처 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Dummy (최빈 클래스) | Dummy | 기준선 v1.0.1, 최빈 클래스(2등급)만 예측 | 0.241744 ±0.000004 | 0.568913 | 0.000000 | 0 | 69,774 | -0.446943 | S1 |
| LR 기본 C=1 (무가중치) ★ | Logistic Regression | v1.0.1 전처리 + lbfgs, C=1, max_iter=1500; 개인 최종 모델 | 0.683677 ±0.001071 | 0.739495 | 0.638433 | 467 | 24,761 | -0.005010 | S1 |
| LR class_weight=balanced | Logistic Regression | 클래스 가중치 balanced | 0.665166 ±0.001406 | 0.691064 | 0.742024 | 2,445 | 15,555 | -0.023522 | S1 |
| LR + Random Under Sampling | Logistic Regression | 학습 Fold 내 RandomUnderSampler(seed 42), 검증 비변경 | 0.646302 ±0.000997 | 0.669517 | 0.731490 | 2,701 | 16,034 | -0.042386 | S1 |
| LR RandomOverSampler | Logistic Regression | 학습 Fold 내 단순 무작위 복제 오버샘플링 | 0.663369 ±0.001730 | 0.689510 | 0.741580 | 2,493 | 15,538 | -0.025319 | S2 |
| LR 제약형 SMOTE + 복제 fallback | Logistic Regression | 동일 범주·층수·가구수·연령 상태 안에서만 수치 3개 보간 (stock SMOTENC 아님) | 0.663206 ±0.001461 | 0.689356 | 0.741379 | 2,491 | 15,554 | -0.025481 | S2 |
| LR C=0.1 (제한 튜닝) | Logistic Regression | 규제 강도 C=0.1 | 0.669949 ±0.002948 | 0.736790 | 0.623026 | 309 | 25,994 | -0.018739 | S1 |
| LR C=10 (제한 튜닝) | Logistic Regression | 규제 강도 C=10 | 0.683152 ±0.001903 | 0.736454 | 0.642646 | 562 | 24,372 | -0.005536 | S1 |
| DT 초기 | Decision Tree | max_depth=12, min_samples_leaf=5 | 0.594238 ±0.004512 | 0.692882 | 0.515708 | 165 | 33,626 | -0.094450 | S1 |
| DT balanced | Decision Tree | 클래스 가중치 balanced | 0.598070 ±0.008850 | 0.622515 | 0.678405 | 2,499 | 19,940 | -0.090618 | S1 |
| RF 초기 (max_features=sqrt) | Random Forest | 100 trees, depth 20, leaf 2, max_features=sqrt | 0.357704 ±0.015771 | 0.611531 | 0.122309 | 2 | 61,238 | -0.330983 | S1 |
| RF (max_features=0.1) | Random Forest | 후보 변수 수 제한 완화 (max_features=0.1) | 0.628992 ±0.003829 | 0.708375 | 0.489681 | 187 | 35,420 | -0.059696 | S1 |
| RF balanced (max_features=0.1, leaf=2) | Random Forest | 클래스 가중치 balanced | 0.647867 ±0.000646 | 0.672554 | 0.694256 | 1,862 | 19,471 | -0.040820 | S1 |
| RF balanced mf=0.1 / leaf=1 (제한 튜닝) | Random Forest | class_weight=balanced, max_features=0.1, min_samples_leaf=1 | 0.651538 ±0.001028 | 0.678075 | 0.694800 | 1,662 | 19,633 | -0.037149 | S1 |
| RF balanced mf=0.2 / leaf=1 (제한 튜닝) | Random Forest | class_weight=balanced, max_features=0.2, min_samples_leaf=1 | 0.658667 ±0.000974 | 0.685932 | 0.679866 | 1,516 | 20,821 | -0.030020 | S1 |
| RF balanced mf=0.2 / leaf=2 (제한 튜닝) | Random Forest | class_weight=balanced, max_features=0.2, min_samples_leaf=2 | 0.656677 ±0.001355 | 0.682742 | 0.685198 | 1,673 | 20,292 | -0.032011 | S1 |
| GBC 초기 (max_features=sqrt) | sklearn GradientBoostingClassifier | 100 stages, lr 0.1, depth 3, max_features=sqrt | 0.469489 ±0.016769 | 0.649635 | 0.253819 | 47 | 52,017 | -0.219198 | S1 |
| GBC (max_features=None) | sklearn GradientBoostingClassifier | 후보 변수 수 제한 완화 (max_features=None) | 0.612109 ±0.005118 | 0.695506 | 0.515593 | 220 | 33,579 | -0.076578 | S1 |
| GBC sample_weight (max_features=None) | sklearn GradientBoostingClassifier | 학습 Fold sample_weight(균형 가중) | 0.604893 ±0.002464 | 0.630857 | 0.675796 | 3,118 | 19,503 | -0.083794 | S1 |

★ 개인 최종 모델로 동결된 설정(LR C=1). 개발 CV 최고 평균으로 선택되었고 홀드아웃은 1회 평가되었다(§6). Accuracy·Recall3·3→1·3→2는 `personal_reference_snapshot.json`의 혼동행렬(행=실제)에서 산출했고, 12개 설정(`model_summary.csv` 8 + `followup_model_summary.csv` 4)은 해당 CSV의 Accuracy·Recall3과 일치함을 확인했다(JSON `derived_fields`).

### 한 요소 비교 짝 (같은 Fold, 차이 = 처리 − 무처리 기준)

| 모델 | 무처리 기준 | 처리 | ΔMacro F1 | Δ3→1 건수 | Δ3→2 건수 | 비고 |
|---|---|---|---:|---:|---:|---|
| LR | LR 기본 C=1 (무가중치) | LR class_weight=balanced | -0.018511 | +1,978 | -9,206 | 클래스 가중치 |
| LR | LR 기본 C=1 (무가중치) | LR + Random Under Sampling | -0.037375 | +2,234 | -8,727 | Random Under Sampling |
| LR | LR 기본 C=1 (무가중치) | LR RandomOverSampler | -0.020308 | +2,026 | -9,223 | Random Over Sampling(복제) |
| LR | LR 기본 C=1 (무가중치) | LR 제약형 SMOTE + 복제 fallback | -0.020471 | +2,024 | -9,207 | 제약형 SMOTE+복제 (stock SMOTENC 아님) |
| DT | DT 초기 | DT balanced | +0.003832 | +2,334 | -13,686 | 클래스 가중치 |
| RF | RF (max_features=0.1) | RF balanced (max_features=0.1, leaf=2) | +0.018876 | +1,675 | -15,949 | 클래스 가중치 (max_features=0.1 고정) |
| GBC | GBC (max_features=None) | GBC sample_weight (max_features=None) | -0.007216 | +2,898 | -14,076 | sample_weight (max_features=None 고정) |

- 가중치·재표집은 대체로 3→2를 줄이고 3→1을 늘린다(RF·DT·GBC·LR 모두 3→1 증가). Macro F1이 올라도 3→1이 크게 늘었다는 점을 함께 본다. 이는 이 표의 관측이며 인과·일반화 주장이 아니다.
- 일반 SMOTENC는 지역 코드 조합·특수 연령(0/995) 보간 문제로 안전성 파일럿에서 불채택했고 **전체 CV를 실행하지 않았다**. 위 제약형 SMOTE는 별도 자체 구현이다.
- 같은 설정을 한 줄에 놓았지만 RF/GBC의 "초기"(max_features=sqrt)와 후속 설정은 하이퍼파라미터 예산이 함께 달라 한 요소 비교가 아니다. 한 요소 비교는 위 짝 표를 본다.

## 4. 표2 — 후속 연구의 층화 결과

T1~T14의 학습 설정과 T15의 사후 규칙 파생값 중 층화 seed 42 5-Fold 값이 있는 설정이다. LR-C와 LGBM-O 등 비교 기준으로 재사용된 팔은 원 실험(T1, T4)에 한 번만 적었다. 근거 집계 JSON/CSV와 위치는 부록과 JSON `source_location`에 있다.

| 실험 | 설정 | 모델 계열 | 변경 요소 | CV Macro F1 (±SD) | Accuracy | Recall3 | 3→1 | 3→2 | Δ(LR-C) | 출처 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| **T1 — 연령 995 처리(LR)** | | | | | | | | | | |
| T1 | A: age 원값 유지 (T0 LR C=1과 동일 값, 중복) | Logistic Regression (C=1) | T0 LR C=1과 동일 설정 (연령 원값, 표시 없음) | 0.683677 ±0.001071 | 0.739495 | 0.638433 | 467 | 24,761 | -0.005010 | S3, S4 |
| T1 | B: 원값 + age_is_995 표시 | Logistic Regression (C=1) | age_is_995 표시 1열 추가 | 0.687129 ±0.001455 | 0.741208 | 0.639823 | 438 | 24,693 | -0.001559 | S3, S4 |
| T1 | C (LR-C): 995만 학습 Fold 중앙값 대체 + 표시 ◆기준 | Logistic Regression (C=1) | 995만 학습 Fold 비995 중앙값(15.0)으로 대체 + age_is_995 표시; 이후 연구의 기준선 | 0.688687 ±0.001508 | 0.741831 | 0.639794 | 452 | 24,681 | +0.000000 | S3, S4 |
| T1 | D: 학습 age=995 행 삭제 | Logistic Regression (C=1) | 학습 Fold에서만 age==995 행 제거(검증은 원값) | 0.686523 ±0.001348 | 0.739313 | 0.642130 | 449 | 24,521 | -0.002165 | S3, S4 |
| T1 | E1: 995만 q99 절단 | Logistic Regression (C=1) | 학습 Fold 비995 q99(=90)로 995만 대체, 표시 없음 | 0.687526 ±0.001318 | 0.741304 | 0.640081 | 453 | 24,660 | -0.001161 | S3, S4 |
| T1 | E2: 995 q99 절단 + 표시 | Logistic Regression (C=1) | E1 + 원본995 표시 | 0.688664 ±0.001265 | 0.741812 | 0.640152 | 455 | 24,653 | -0.000024 | S3, S4 |
| T1 | E3: 전체 연령 q99 절단 | Logistic Regression (C=1) | 학습 Fold q99 초과 연령 모두 절단 | 0.687899 ±0.001360 | 0.741548 | 0.640496 | 452 | 24,632 | -0.000788 | S3, S4 |
| T1 | F1: 학습 age>55 삭제 | Logistic Regression (C=1) | 학습 Fold에서 age>55 행 제거 | 0.685903 ±0.001126 | 0.737577 | 0.645785 | 442 | 24,273 | -0.002785 | S3, S4 |
| T1 | F2: FAITH 4조건 OR 삭제 | Logistic Regression (C=1) | 학습 age>55 OR floors>5 OR area>20 OR height>10 제거 | 0.684010 ±0.001922 | 0.736804 | 0.643076 | 449 | 24,455 | -0.004678 | S3, S4 |
| **T3 — 지역·구조·연령 조합(LR)** | | | | | | | | | | |
| T3 | G1: C + 연령 8구간 | Logistic Regression (C=1) | 원본 연령 8구간 원핫 추가 | 0.690076 ±0.002062 | 0.742421 | 0.640496 | 454 | 24,630 | +0.001389 | S5, S6 |
| T3 | G2: G1 + 1차 지역×기초구조 | Logistic Regression (C=1) | geo1×foundation_type 교차 원핫 추가 | 0.690993 ±0.002655 | 0.742714 | 0.640081 | 478 | 24,635 | +0.002306 | S5, S6 |
| T3 | G3: G1 + 1차 지역×연령구간 | Logistic Regression (C=1) | geo1×연령구간 교차 원핫 추가 | 0.689817 ±0.002214 | 0.742205 | 0.640009 | 457 | 24,661 | +0.001129 | S5, S6 |
| T3 | G4: G1 + 기초구조×연령구간 | Logistic Regression (C=1) | foundation_type×연령구간 교차 원핫 추가 | 0.690266 ±0.002258 | 0.742503 | 0.640941 | 456 | 24,597 | +0.001579 | S5, S6 |
| T3 | G5: G1 + 세 교차항 모두 | Logistic Regression (C=1) | 위 세 가지 2변수 조합을 동시에 추가 | 0.690569 ±0.002884 | 0.742215 | 0.639508 | 483 | 24,670 | +0.001882 | S5, S6 |
| **T4 — XGBoost/LightGBM 비교** | | | | | | | | | | |
| T4 | XGBoost A (원핫, 연령 원값) | XGBoost | hist, depth 6, lr 0.05, 300 rounds; 입력 A(연령 원값) | 0.637946 ±0.004431 | 0.711982 | 0.543799 | 238 | 31,593 | -0.050741 | S7 |
| T4 | XGBoost C (원핫, 995 대체+표시) | XGBoost | 입력 C(995 중앙값 대체+표시); 나머지 XGB_A와 동일 | 0.638558 ±0.004235 | 0.712318 | 0.544186 | 239 | 31,565 | -0.050129 | S7 |
| T4 | LightGBM A (원핫, 연령 원값) | LightGBM | num_leaves 31, lr 0.05, 300 rounds; 입력 A | 0.669830 ±0.003634 | 0.732603 | 0.589833 | 297 | 28,322 | -0.018857 | S7 |
| T4 | LightGBM C = LGBM-O (원핫) | LightGBM | 입력 C; 이후 연구에서 "LGBM-O"(원핫 LightGBM) 기준으로 재사용 | 0.669847 ±0.002892 | 0.732636 | 0.590077 | 303 | 28,299 | -0.018840 | S7 |
| **T6R — 지역 변수 제거 ablation(독립 재현)** | | | | | | | | | | |
| T6R | R12: 지역 geo3 제거 (geo1·geo2 유지) | Logistic Regression (C=1) | C에서 geo_level_3_id 원핫 제거 | 0.653098 ±0.002070 | 0.718189 | 0.599550 | 326 | 27,615 | -0.035589 | S8 |
| T6R | R1: geo1만 유지 | Logistic Regression (C=1) | C에서 geo2·geo3 제거 | 0.593425 ±0.004769 | 0.671786 | 0.521254 | 217 | 33,187 | -0.095262 | S8 |
| T6R | R0: 지역 3열 모두 제거 | Logistic Regression (C=1) | C에서 지역 3열 제거 | 0.457895 ±0.002635 | 0.589414 | 0.174248 | 358 | 57,258 | -0.230793 | S8 |
| **T7 — 지역 계층적 축소** | | | | | | | | | | |
| T7 | T7 H3_050 | Logistic Regression (C=1) | geo3 원핫 블록 가중 s=0.5 (축소) | 0.686370 ±0.001933 | 0.742383 | 0.635753 | 383 | 25,032 | -0.002318 | S9 |
| T7 | T7 H3_025 | Logistic Regression (C=1) | geo3 블록 s=0.25 | 0.677258 ±0.001883 | 0.737514 | 0.627325 | 340 | 25,663 | -0.011429 | S9 |
| T7 | T7 H3_010 | Logistic Regression (C=1) | geo3 블록 s=0.10 | 0.660943 ±0.002187 | 0.724448 | 0.607877 | 327 | 27,033 | -0.027744 | S9 |
| T7 | T7 H23_050 | Logistic Regression (C=1) | geo2+geo3 블록 s=0.5 | 0.684226 ±0.001340 | 0.742220 | 0.633144 | 362 | 25,235 | -0.004461 | S9 |
| T7 | T7 H23_025 | Logistic Regression (C=1) | geo2+geo3 블록 s=0.25 | 0.669530 ±0.002541 | 0.736138 | 0.619500 | 262 | 26,287 | -0.019157 | S9 |
| T7 | T7 H23_010 | Logistic Regression (C=1) | geo2+geo3 블록 s=0.10 | 0.637978 ±0.002677 | 0.714965 | 0.582595 | 217 | 28,907 | -0.050709 | S9 |
| **T8 — LightGBM native categorical** | | | | | | | | | | |
| T8 | LGBM-N: 범주형 11열 native categorical | LightGBM | LGBM-O(원핫)와 같은 파라미터에서 범주형 11열만 학습 Fold 어휘의 native categorical로 변경 | 0.699342 ±0.003306 | 0.746450 | 0.647648 | 530 | 24,055 | +0.010654 | S10 |
| **T9 — LightGBM 부분 native** | | | | | | | | | | |
| T9 | LGBM-P: 텍스트 범주 8열만 native | LightGBM | geo 3열은 원핫 유지, 텍스트 범주 8열만 native categorical | 0.669966 ±0.003613 | 0.732540 | 0.590349 | 300 | 28,283 | -0.018721 | S11 |
| **T10 — 지역 계층적 Target Encoding** | | | | | | | | | | |
| T10 | LR_TE: 지역 계층적 Target Encoding | Logistic Regression (C=1) | 지역 3열 원핫을 학습 Fold 내부 5-Fold OOF 계층적 TE 9특성(m=20)으로 교체 | 0.667167 ±0.002463 | 0.735145 | 0.629088 | 359 | 25,521 | -0.021520 | S12 |
| T10 | LGBM_TE: 지역 계층적 Target Encoding | LightGBM | LGBM-O의 지역 원핫을 계층적 TE 9특성으로 교체 | 0.688170 ±0.003239 | 0.742417 | 0.639049 | 387 | 24,798 | -0.000518 | S12 |
| **T12 — 구조 파생 특성** | | | | | | | | | | |
| T12 | LR_S | Logistic Regression (C=1) | 구조 요약 4특성(재료·용도 개수) 추가 | 0.688845 ±0.001353 | 0.741894 | 0.640396 | 463 | 24,628 | +0.000157 | S13 |
| T12 | LR_D | Logistic Regression (C=1) | 크기·비율 3특성(층당 높이, 면적×층수, 층당 가구수) 추가 | 0.688817 ±0.001546 | 0.741707 | 0.639909 | 456 | 24,669 | +0.000129 | S13 |
| T12 | LR_SD | Logistic Regression (C=1) | S+D 7특성 추가 | 0.688843 ±0.001417 | 0.741721 | 0.640425 | 456 | 24,633 | +0.000156 | S13 |
| T12 | LGBM_S | LightGBM | LGBM-O + 구조 요약 4특성 | 0.670341 ±0.003298 | 0.732948 | 0.589288 | 307 | 28,350 | -0.018346 | S13 |
| T12 | LGBM_D | LightGBM | LGBM-O + 크기·비율 3특성 | 0.670550 ±0.003257 | 0.732771 | 0.590378 | 300 | 28,281 | -0.018137 | S13 |
| T12 | LGBM_SD | LightGBM | LGBM-O + S+D 7특성 | 0.670725 ±0.003266 | 0.733097 | 0.590535 | 306 | 28,264 | -0.017963 | S13 |
| **T14 — TE·ROS RandomForest 로컬 재현** | | | | | | | | | | |
| T14 | RF + geo1/2/3 OOF TE + 지역 빈도 + 학습 Fold ROS (원본 raw age 유지) | Random Forest | TE(내부 5-Fold, m=20) + 지역 빈도 + 학습 Fold RandomOverSampler + RF; 모델 77열. TE·ROS·RF·age995 원값을 동시에 변경(요인 미분리) | 0.700427 ±0.001070 | 0.739884 | 0.673016 | 744 | 22,071 | +0.011740 | S14, S15 |
| **T15 — 사후 결정 규칙 적용값(참고; 학습 설정 아님)** | | | | | | | | | | |
| T15 | LR_C + 3등급 로짓 오프셋(보정 후) | Logistic Regression (C=1) | T1 C(LR-C) 저장 OOF에 3등급 확률 exp(b) 배 규칙 적용(b는 내부 OOF로 선택, 층화에서 0.1~0.2); 바깥 모델 재학습 없음 | 0.689353 | — | — | — | — | +0.000665 | S16 |
| T15 | LGBM_O + 3등급 로짓 오프셋(보정 후) | LightGBM | T4 LGBM_C_v1(LGBM-O) 저장 OOF에 3등급 확률 exp(b) 배 규칙 적용(b는 내부 OOF로 선택, 층화에서 0.1~0.2); 바깥 모델 재학습 없음 | 0.674168 | — | — | — | — | -0.014519 | S16 |
| T15 | LGBM_TE_G + 3등급 로짓 오프셋(보정 후) | LightGBM | T10 LGBM_TE(층화에서 LGBM_TE_G와 동일) 저장 OOF에 3등급 확률 exp(b) 배 규칙 적용(b는 내부 OOF로 선택, 층화에서 0.1~0.2); 바깥 모델 재학습 없음 | 0.689884 | — | — | — | — | +0.001197 | S16 |

◆ LR-C = T1 C. T1 A는 T0 LR C=1과 같은 설정·같은 값이라 한 번만 센다. 60행 = 학습 고유 설정 56 + T1 A(중복) 1 + T15 사후 규칙 파생값 3.

행별 주의:
- **T1** D/F1/F2는 검증의 age=995 1,125건을 모두 3등급으로 예측했다(1/2등급 구분 불능). 3→1이 줄어 보이는 것은 안전성 개선이 아니다.
- **T3** G2의 Macro F1은 LR-C보다 높지만 3→1은 452→478건으로 늘었다. 사전 기준(F1 개선+3→1 증가 없음)을 만족한 후보는 없었다.
- **T4** LGBM_C_v1이 이후 연구의 "LGBM-O(원핫 LightGBM)"이다. LR_A/LR_C는 T1 A/C 재사용이라 중복하지 않았다.
- **T6R** 원래 T6의 비공개 증거는 회수하지 못했다(`BLOCKED`). R12/R1/R0는 같은 설정의 별도 재현이며 원래 T6 결과를 검증하거나 대체하지 않는다.
- **T8/T9** N은 층화에서 0.6993이지만 geo2 Group에서는 LR-C보다 낮았다(0.5457, T8 보고서). P는 층화에서 LGBM-O와 거의 같아 N의 이득을 유지하지 못했다(T9 보고서). 지역 일반화는 이 표에서 판단하지 않는다. LGBM-O는 T4 값을 재사용했다.
- **T10** `LGBM_TE`의 층화 값은 LR-C와 비슷하지만 Group CV에서는 학습·검증 TE 분포 차이로 해석이 제한된다(T10 보고서). T11의 `LR_TE_G`/`LGBM_TE_G` 층화 값은 T10 값을 그대로 재사용한 것이라 따로 적지 않았다.
- **T12** 여섯 변형 모두 사전 기준(+0.003) 미달이다.
- **T14** 같은 층화 Fold에서 0.700427이 혼동행렬 셀 단위까지 재현되었다(Stage A EXACT). TE·ROS·RF·연령 원값 유지를 한꺼번에 바꿔 요인은 분리되지 않았고, 처음 보는 2차 지역(geo2 Group 20 Fold)에서는 LR-C(−0.0088), LGBM-O(−0.0250)보다 낮았다. 층화에서만 0.7을 넘는다.
- **T15** 모델 재학습 없이 3등급 확률에 exp(b)를 곱하는 규칙을 바깥 Fold별로 선택·적용한 결과다. 층화는 G4(해롭지 않음 확인용 감시)에 쓰인 보조 지표이고 보고서는 이를 낙관적이라고 부른다. 값은 보정 전 평균 + `stratified_mean_delta`로 계산했으며 보고서 표기 4자리와 일치한다. Fold 표준편차·Accuracy·Recall3·3→1/3→2는 집계에 없어 `—`이다.

### 같은 Fold 집합에서의 관측 순위 (학습된 설정만, CV Macro F1 상위 10, 선택 통계)

| 순위 | 실험 | 설정 | CV Macro F1 | Δ(LR-C) | 3→1 | Recall3 |
|---:|---|---|---:|---:|---:|---:|
| 1 | T14 | RF + geo1/2/3 OOF TE + 지역 빈도 + 학습 Fold ROS (원본 raw age 유지) | 0.700427 | +0.011740 | 744 | 0.673016 |
| 2 | T8 | LGBM-N: 범주형 11열 native categorical | 0.699342 | +0.010654 | 530 | 0.647648 |
| 3 | T3 | G2: G1 + 1차 지역×기초구조 | 0.690993 | +0.002306 | 478 | 0.640081 |
| 4 | T3 | G5: G1 + 세 교차항 모두 | 0.690569 | +0.001882 | 483 | 0.639508 |
| 5 | T3 | G4: G1 + 기초구조×연령구간 | 0.690266 | +0.001579 | 456 | 0.640941 |
| 6 | T3 | G1: C + 연령 8구간 | 0.690076 | +0.001389 | 454 | 0.640496 |
| 7 | T3 | G3: G1 + 1차 지역×연령구간 | 0.689817 | +0.001129 | 457 | 0.640009 |
| 8 | T12 | LR_S | 0.688845 | +0.000157 | 463 | 0.640396 |
| 9 | T12 | LR_SD | 0.688843 | +0.000156 | 456 | 0.640425 |
| 10 | T12 | LR_D | 0.688817 | +0.000129 | 456 | 0.639909 |

이 순위는 학습된 56개 고유 설정(60행 중 T15 사후 규칙 파생 3행과 T0와 중복인 T1 A를 제외)을 같은 검증 Fold로 반복 비교해 나온 값이다. 상위 설정끼리의 차이는 선택 편향과 Fold 간 변동(상위 설정의 SD는 약 0.001~0.003)에 비해 작은 경우가 많고, 3→1·Recall3·지역 일반화 결과가 다르다. 표의 어떤 설정도 이 문서만으로 채택하거나 "최고"라고 하지 않는다.

#### 사후 결정 규칙 적용값 (참고, 순위에서 제외)

T15 세 행은 새로 학습한 설정이 아니라 기존 OOF에 3등급 확률 보정 규칙을 사후 적용한 파생 평균이다(보정 전 평균 + `stratified_mean_delta`). 순위에 넣지 않고 보정 전 값과 나란히만 둔다.

| 대상 | 보정 전(층화 CV 평균) | 보정 후(참고) | 차이 |
|---|---:|---:|---:|
| LR_C | 0.688687 | 0.689353 | +0.000665 |
| LGBM_O | 0.669847 | 0.674168 | +0.004321 |
| LGBM_TE_G | 0.688170 | 0.689884 | +0.001715 |

## 5. 표3 — 층화 결과가 없는 실험

각 보고서와 집계 JSON에서 "stratified"·"층화" 언급 수를 스캔했다(JSON `no_stratified_result`). T16~T27은 층화 언급이 보고서·집계 모두 0건이며 geo2 Group 분할(seed 42/101/202/303)만 쓴다.

| 실험 | 이유 | 관련 보고(링크) | 스캔(층화 언급 md/json) |
|---|---|---|---|
| T2 오류 진단 | 신규 설정 없음: T1의 기존 OOF(C·E2 등)를 재분석한 진단(학습 0회) | [AGE995_DIAGNOSTICS_20261014.md](AGE995_DIAGNOSTICS_20261014.md) | 0/0 (geo2 언급 0) |
| T5 지역 Group CV | 신규 학습은 geo2/geo1 Group CV 전용(지역 일반화). 층화 값은 T1 C·T3 G2의 기존 값을 인용만 함 | [GEO_GROUP_CV_20261014.md](GEO_GROUP_CV_20261014.md) | 29/6 (geo2 언급 3) |
| T6S C 보완 감사 | Group CV의 C 행 단위 보완 감사. 층화 값은 T6R 표와 동일(중복) | [GEO_ABLATION_C_GROUP_SUPPLEMENT_20261014.md](GEO_ABLATION_C_GROUP_SUPPLEMENT_20261014.md) | 1/41 (geo2 언급 4) |
| T11 그룹 인지 내부 Fold TE | 신규 학습은 geo2/geo1 전용. 층화 값(LR_TE_G 0.6672, LGBM_TE_G 0.6882)은 T10 값 재사용(내부 분할이 같아 새 층화 학습 없음) | [GEO_TE_GROUP_INNER_20261014.md](GEO_TE_GROUP_INNER_20261014.md) | 8/43 (geo2 언급 19) |
| T13 LGBM_TE_G geo2 확인 | geo2 Group 새 seed 101/202/303 전용. 층화 값은 T10 재사용 언급만 | [GEO2_TE_CONFIRM_20261014.md](GEO2_TE_CONFIRM_20261014.md) | 1/1 (geo2 언급 9) |
| T6 원본(R12/R1/R0 원래 실행) | 원래 T6 비공개 증거 미확보(BLOCKED_T6_PRIVATE_EVIDENCE_MISSING). 같은 설정의 독립 재현 T6R만 표2에 포함 | [GEO_ABLATION_RETRIEVAL_20261014.md](GEO_ABLATION_RETRIEVAL_20261014.md) | 5/3 (geo2 언급 0) |
| T16 RF 요인 분리 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [RF_FACTORS_20261015.md](RF_FACTORS_20261015.md) | 0/0 (geo2 언급 4) |
| T17 3등급 규칙 고정 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [GRADE3_RULE_FREEZE_20261015.md](GRADE3_RULE_FREEZE_20261015.md) | 0/0 (geo2 언급 2) |
| T18 LGBM 내부 튜닝 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [LGBM_NESTED_TUNING_20261015.md](LGBM_NESTED_TUNING_20261015.md) | 0/0 (geo2 언급 1) |
| T19 고정 앙상블 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [FIXED_ENSEMBLE_20261015.md](FIXED_ENSEMBLE_20261015.md) | 0/0 (geo2 언급 1) |
| T20 LGBM 3등급 가중 1.5 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [LGBM_GRADE3_WEIGHT_20261015.md](LGBM_GRADE3_WEIGHT_20261015.md) | 0/0 (geo2 언급 1) |
| T21 연령 곱·구조 조합 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [FEATURE_INTERACTIONS_20261015.md](FEATURE_INTERACTIONS_20261015.md) | 0/0 (geo2 언급 1) |
| T22 재료쌍·구조별 층수 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [MATERIAL_STRUCTURE_20261015.md](MATERIAL_STRUCTURE_20261015.md) | 0/0 (geo2 언급 1) |
| T23 누적 이진·단조 투영 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [ORDINAL_DAMAGE_20261015.md](ORDINAL_DAMAGE_20261015.md) | 0/0 (geo2 언급 1) |
| T24 상위 지역 맥락 6열 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [GEO_CONTEXT_20261015.md](GEO_CONTEXT_20261015.md) | 0/0 (geo2 언급 2) |
| T25 CatBoost (C 입력) | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [CATBOOST_C_20261015.md](CATBOOST_C_20261015.md) | 0/0 (geo2 언급 1) |
| T26 CatBoost 범주형 직접 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [CATBOOST_NATIVE_C_20261015.md](CATBOOST_NATIVE_C_20261015.md) | 0/0 (geo2 언급 1) |
| T27 넓은 파생변수 | geo2 Group 분할 seed 42/101/202/303 전용(층화 분할 결과 없음) | [WIDE_FEATURES_20261015.md](WIDE_FEATURES_20261015.md) | 0/0 (geo2 언급 21) |
| A1 T15~T27 종합 | 신규 학습 0회 종합 분석(geo2 중심). 층화는 T15 G4 조건 언급뿐이며 새 층화 수치 없음 | [EXPERIMENT_SYNTHESIS_20261016.md](EXPERIMENT_SYNTHESIS_20261016.md) | 1/3 (geo2 언급 3) |
| T28 3→1 비용 결정 규칙 | 미승인·미구현·미실행(설계 초안) | [T28_GRADE3_SAFETY_PLAN_DRAFT.md](../docs/T28_GRADE3_SAFETY_PLAN_DRAFT.md) | 1/— (geo2 언급 8) |

- T5·T6S·T11·T13처럼 "층화"가 언급되는 보고서는 이전 층화 값(T1 C, T3 G2, T6R, T10)을 비교용으로 인용·재사용한 것이며 새 층화 학습이 없다.
- T16~T27의 geo2 결과(미관측 2차 지역 일반화)는 층화 평가와 질문이 다르다. 심화 항목으로 해당 보고서를 따로 본다.
- T2 오류 진단은 학습 0회 분석이고, T28은 설계 초안이다.

## 6. 홀드아웃 주의

| 설정 | 홀드아웃 행 | Macro F1 | Accuracy | 평가 횟수 | 출처 |
|---|---:|---:|---:|---:|---|
| T0 LR 기본 C=1 (무가중치) | 52,121 | 0.686771 | 0.742369 | 1 | reports/personal_final_holdout.json |

- 과거 개인 홀드아웃 52,121행은 **T0 LR C=1 한 설정에만, 한 번** 사용했다(동결 후 평가, 재튜닝 없음). 위 값 하나만 홀드아웃 값이며, 표1·표2의 다른 모든 값은 개발 CV다.
- 이 홀드아웃은 이미 결과가 노출된 자료다. 후속 선택·튜닝·early stopping·재평가에 재사용하지 않으며, 새로운 독립 성능으로도 주장하지 않는다. 별도의 최종 평가 설계는 사용자 결정 사항이다.
- 이 문서 작성 과정에서도 홀드아웃 경로에 접근하지 않았다.

## 7. 한계

1. **선택 통계**: 학습된 고유 설정 56개(60행 중 T1 A는 T0와 중복, T15 3행은 사후 규칙 파생값)를 같은 5개 검증 Fold로 반복 비교했다. 상위 설정의 CV 평균은 낙관적일 수 있고 독립 성능이 아니다. 반복 사용한 Fold는 T15 보고서도 낙관적이라고 적었다.
2. **통계 검정 없음**: Fold 5개는 독립 표본이 아니며 표준편차·Fold 승수를 유의성으로 해석하지 않는다.
3. **일치 조건의 한계**: 모든 행이 같은 Fold이지만 전처리·모델·하이퍼파라미터·입력 표현이 서로 다르다. Δ(LR-C)는 단일 요인 효과가 아니다(예: T14, T4).
4. **오류 비용**: 3→1은 가장 중대한 과소예측이지만 Macro F1은 이를 반영하지 않는다. 팀의 오류 비용과 허용 기준은 별도 결정 사항이다.
5. **지역 일반화 미반영**: 층화 분할은 같은 지역이 학습·검증에 모두 나온다. 처음 보는 지역에 대한 성능은 geo2/geo1 Group 보고서(표3)를 따른다. 또 지진 강도·지반 조건 등은 반영하지 않았고 인과·안전 판정을 뜻하지 않는다.
6. **재사용·재현**: T0의 과거 8개 원래 OOF는 미확보(snapshot 집계로 검증)이며 기존 LR C=1은 새 OOF로 재현했다. T1 A/B/C, LR-C 등은 이전 OOF 재사용이다. 재사용 여부는 JSON `origin`에 있다.
7. **보고서 표기 차이**: T12 LR_S(0.68885 vs 원자료 0.688845), T12 LGBM_SD(0.67073 vs 0.670725), T6R R0(0.45790 vs 0.457895)는 보고서 표기가 원자료 반올림과 마지막 자리에서 1 다르다(이중 반올림 추정). 이 문서는 원자료를 사용했다.
8. **미포함**: T14 원본 탐색의 LR/DT/XGB/LGBM 값은 Fold 0 한정이라 5-Fold가 아니어서 제외했다. T0의 "LR C=1 재현 점검"은 새 설정이 아니라 중복이다.
9. 이 문서는 독립 검토 PASS_WITH_NOTES 반영본이며, 팀 공통 분할·팀 모델 선정·main 통합과 무관하다.

## 부록 — 출처 파일과 SHA-256

경로는 저장소 루트 기준이다. 각 행의 정확한 위치(키/행 조건)는 JSON `source_location`이다. 해시는 작성 시점 작업 트리 파일의 바이트 SHA-256이다.

| ID | 파일 | SHA-256 |
|---|---|---|
| S1 | `reports/personal_reference_snapshot.json` | `35b6a6ba6e740353e3dd5fc72217377341dc15e8fcdf5a9cba4a0a7270cf12b6` |
| S2 | `reports/personal_sampling_summary.csv` | `e3aa2a533a354272b988efbbcb8d367eabc5b1205db1b536dd36ecae18ab612c` |
| S3 | `reports/age995_followup_v1/model_summary.csv` | `4e8410a06d9502ce348b820df46e7b00a766304de5f2d08ccc2f0a4a4e41aef9` |
| S4 | `reports/age995_followup_v1/pooled_group_summary.csv` | `083da0e475e2e096bc860b02dd0777c0bad3d2c60ae8d810dc7c05dfd48765d9` |
| S5 | `reports/age995_interactions_v1/models.csv` | `34b473b52fad285289d9f1c1ab75cac5f60dd16d0c5b491572c20d2db02afaff` |
| S6 | `reports/age995_interactions_v1/pooled_groups.csv` | `14cab135fe762973b897c647b9502375ffc2bb1895f84b9cfe95811d7a876c0a` |
| S7 | `reports/model_families_v1/model_summary.csv` | `45e5964f249ae3bbefdf1e3cfdf61f79a72f51fd724078ef9dec65e271bac395` |
| S8 | `reports/geo_ablation_local_repro_20261014.json` | `aaf26a955438fa0a3fc79bbdb32e9e8f641d765edae34ff1741742441108666f` |
| S9 | `reports/geo_hier_shrink_20261014.json` | `1eb7b754eddca70a4d6680d65b9d875caa5ff4173b4d97f87f08ea2390503f9a` |
| S10 | `reports/lgbm_native_cat_20261014.json` | `e5d86b30dd226f70c7c928e462ee981d739b9695afcf70a6893ce6d28b4764bc` |
| S11 | `reports/lgbm_partial_native_20261014.json` | `099c4d2d702559b2f8953b91a2b2efd845677668765f792ae431fa7555ec606c` |
| S12 | `reports/geo_hier_te_20261014.json` | `92785b94c7dc770a6e31f07585ac1164199229a54792b4c2e3d2cbeb2d711e93` |
| S13 | `reports/struct_features_20261014.json` | `fe4a133ef4878633e8ee1d1e2800ebe18e9f466507caccf79c7594d060785351` |
| S14 | `reports/te_ros_rf_repro_20261015.json` | `441ddde5e5883da831206df56d2b3b92e1bcec378c05ddba033fb6e540f07c2a` |
| S15 | `reports/te_ros_rf_exploratory_summary.json` | `66f8d74f90beb06cc89b5dcfcd2050b55585537af1dfefaed43edc269ac7670d` |
| S16 | `reports/grade3_decision_rule_20261015.json` | `3aa8c191518033dc82a1aa4d32c58016da4c6d8ee49a7775abcd6e2d33fe91e1` |

서술 근거로 확인한 보고서(위 표의 숫자·조건 대조용):

| 파일 | SHA-256 |
|---|---|
| `docs/PERSONAL_MODEL_SELECTION_20261013.md` | `9908fa60fea289d01fda423240170641fad4074b0b919fb6d3e128eb0d8d19f1` |
| `docs/T28_GRADE3_SAFETY_PLAN_DRAFT.md` | `60e83205d44c458a432e52820a61f01adf8d64729ef0dc780a20cb7cdf265465` |
| `reports/AGE995_DIAGNOSTICS_20261014.md` | `a0424600b418f486db669cd38316335e5c7a517d6ceae7a5357f425b516cc5b9` |
| `reports/AGE995_EXPERIMENT_20261013.md` | `1641205970d710340c78b8070aacaa9a207f182bfa635674e275bf14a6452319` |
| `reports/AGE995_FOLLOWUP_20261013.md` | `ba9a5a745b6785dfc4456de17509cfcd96ba72a225b5b963c3946981b2a3c885` |
| `reports/AGE995_INTERACTIONS_20261014.md` | `edfed466ed04a9cb529ed2a74a7431c691419f0d0f476cda1c284aa98be367c1` |
| `reports/BASELINE_REPORT.md` | `aee0dc115ae244164212b75088947a6645ec2fb68b3a014db3b1ee134645fdac` |
| `reports/CATBOOST_C_20261015.md` | `fa4bd38df5af3dc69df6eba9a55707a0af95aab4756047b6a594136984dda569` |
| `reports/CATBOOST_NATIVE_C_20261015.md` | `6ba2b0d55b9ae26c3f19f0d4e38495ab322b9e87afd4fa8452cc40a261e2a398` |
| `reports/EXPERIMENT_SYNTHESIS_20261016.md` | `10920c0327d8c09b97ce911ad0f61e052758dd18e364c4b4129ffc7788e0558b` |
| `reports/FEATURE_INTERACTIONS_20261015.md` | `a674209cb8b8f84c03e81b203e34b77b14f16b5a172ceeb68e5d7d24f2eb145d` |
| `reports/FIXED_ENSEMBLE_20261015.md` | `2a03a3e25c2d3482f6f4d377ba1de08f41ec1d12f57d6593eb927ed5e744104e` |
| `reports/FOLLOWUP_REPORT_20261013.md` | `9b625dfa4d428d3974e5f048759f0f0994e519ce9832b10be7575af43086785b` |
| `reports/GEO2_TE_CONFIRM_20261014.md` | `6f1631b3e739ffbe45c01e94e88909984eb17fb21f72ca6c8fba2c369c797505` |
| `reports/GEO_ABLATION_C_GROUP_SUPPLEMENT_20261014.md` | `1858d36c05f725c76f98afb7e97bbdf059e4f2ee7d484afea251269e0a6fde5c` |
| `reports/GEO_ABLATION_LOCAL_REPRO_20261014.md` | `2f0e0e71f87b3356db4bbef1783754d895ec82178da9e3e88db92b4732478488` |
| `reports/GEO_ABLATION_RETRIEVAL_20261014.md` | `d00e93ac38fc3d8da7dae8ec4c44ec9cbc039d7c1408010c30ad7b1a9a08421a` |
| `reports/GEO_CONTEXT_20261015.md` | `d61a91f46c22b70da6ff7e8cec4d8be77e6fc0d3f1a96b90047b1cf8fca82cc3` |
| `reports/GEO_GROUP_CV_20261014.md` | `323de42e24351ebb88189772c0dd6f745c9858774817c2f97e632f6c933b1c88` |
| `reports/GEO_HIER_SHRINK_20261014.md` | `7c5803a2fd41a02d26a93b8c2da4b544f05fe81a7db6990518f9ed14881e1f4a` |
| `reports/GEO_HIER_TE_20261014.md` | `c65ac1725bb75da174b30b9b3ec946b61ae5973ec805def4a978739f72913d90` |
| `reports/GEO_TE_GROUP_INNER_20261014.md` | `7e589ff32b6c81a411895b969775ec9575bd8aebc259aa324b9f3487721241d1` |
| `reports/GRADE3_DECISION_RULE_20261015.md` | `314548bbede15eecb3ffe366ef79c2e8ea0bfb4b9d7c347e4e79d5f77434f34b` |
| `reports/GRADE3_RULE_FREEZE_20261015.md` | `dc160a4a5082f017c7f313edf042e1e89a25d48bb895919c056a133d94f29732` |
| `reports/LGBM_GRADE3_WEIGHT_20261015.md` | `74088726cec7029eb8f5bf1d64234bf1fde445611894fb95ab94b6b91c36c5eb` |
| `reports/LGBM_NATIVE_CAT_20261014.md` | `2b03218d78ed55d5c4191ced328d68480ffc294bad1b61695492d6e87314f053` |
| `reports/LGBM_NESTED_TUNING_20261015.md` | `d2d1275453c928ae63da367b83bc7ef481ffcf7d6aafb7ffd6230655cb3a4c10` |
| `reports/LGBM_PARTIAL_NATIVE_20261014.md` | `324c2b1f587c746ad94d8b683725b3e0bcdcb573bbf5edb405020bfd03d1a2b0` |
| `reports/MATERIAL_STRUCTURE_20261015.md` | `b9381f3d2d76b596587f7f613909ae014403f876376d026ae9da423163d39c0f` |
| `reports/MODEL_FAMILY_COMPARISON_20261014.md` | `89438b717980c4b5fb55a45176bb66b9ba8f0d90bfd746e300af8221f0fd1278` |
| `reports/ORDINAL_DAMAGE_20261015.md` | `bd85cfeac276c017d65c666bd1e89cfb55f6db262529b62ca1232001fef9c5c3` |
| `reports/PERSONAL_FINAL_REPORT_20261013.md` | `07172c7b2e3e2b5ce8c0f0eeef7bdee234d4af51a5c04a2204ed80d7ab064676` |
| `reports/RF_FACTORS_20261015.md` | `bf81395c9e9eec08abe66c4af2126ea0c94ae6a23a79c32a7dfbcd09fab21741` |
| `reports/STRUCT_FEATURES_20261014.md` | `7c8f94b58bb943dcb7613c9016dfdcba57102b6a4dce1a0f92326eb295ae0651` |
| `reports/TE_ROS_RF_EXPLORATORY_20261014.md` | `d4db0d17c5078262697dd5d3845e09946c9736452fa420b8e3328923bb6ac013` |
| `reports/TE_ROS_RF_REPRO_20261015.md` | `ff5d9e00283cf58bc95255569813cdd0ffc5b6cfb0915594404b0716b3cf2015` |
| `reports/WIDE_FEATURES_20261015.md` | `463447a0107abe0ea5ad748ea8f790b3d0c1e3a22a3bb3591f0812a040f8a642` |
