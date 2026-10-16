# 개인 분석 및 후속 연구 상태 — 2026-10-15

후속 개인 연구는 `worktrees/peter_rules`의 `peter`에서만 진행한다. 새 브랜치/worktree를 만들지 않는다. **T16~T27은 학습 또는 분석·독립 감사·공개 보고 검토 완료다. T26은 T25 대비 평균F1 Δ−0.043886, T27은 LGBM-O 대비 Δ+0.002163이나3→1 오류902→958로 사전 기준 미충족이다. 기존 모델을 유지한다(2026-10-15).** 최신 상태·근거는 [연구 현황](RESEARCH_STATUS.md)을 따른다.

## 남은 작업과 완료 조건

- [x] T27: 20/20·재시도0·독립 감사 PASS_T27·공개 보고 검토959항목 PASS. [보고](../reports/WIDE_FEATURES_20261015.md). 평균F1 0.584218(Δ+0.002163), 양의Fold16/20·seed4/4·Recall3 개선이나 개선폭과3→1 기준 미충족. 추가101종·인코딩 전 논리140종과 실제 행렬 폭을 구분한다.

- [x] T26: 20/20·재시도0·독립 감사 PASS_T26·공개 보고 검토673항목 PASS. [보고](../reports/CATBOOST_NATIVE_C_20261015.md). 평균F1 0.524335, T25 원핫 대비Δ−0.043886·LGBM-O 대비Δ−0.057720, 두 비교 모두 양의Fold0/20·seed0/4·사전 기준 미충족.
- [x] T25: 20회 학습·독립 감사 PASS_T25·공개 보고 검토355항목 PASS. [보고](../reports/CATBOOST_C_20261015.md). 평균 Δ−0.013834, 20개 Fold 모두 기준보다 낮음. 3→1은902→527이나 Recall3 하락으로 기준 미충족.
- [x] T24: 20회 학습·독립 감사 PASS_T24·공개 보고 검토481항목 PASS. [보고](../reports/GEO_CONTEXT_20261015.md). 평균 Δ+0.002072, 3→1 902→912로 다섯 기준 미충족.
- [x] T22: 40/40·독립 감사·공개 보고 검토609항목 PASS. [보고](../reports/MATERIAL_STRUCTURE_20261015.md). 재료쌍 Δ−0.000427/구조별층수 −0.000313, 두 팔 모두 기준 미충족.
- [x] T23: 40이진/20외부단위·독립 감사·공개 보고 검토539항목 PASS. [보고](../reports/ORDINAL_DAMAGE_20261015.md). 평균 Δ−0.000265, 3→1 902→895이나 Recall3 하락으로 기준 미충족.
- [x] 공통 문서 현재성 점검: 현재 규칙·과거 기록 구분, 주장별 근거표·실행 없는 절차 대조. 실험 전체 검증과 구분.
- [x] T16: 80/80·독립 감사·공개 검토 PASS. [보고](../reports/RF_FACTORS_20261015.md). 네 변경 모두 사전 기준 미충족, 중단 원인 미확정.
- [x] T17: 결정 규칙 고정·감사 완료, 신규 fit0. [보고](../reports/GRADE3_RULE_FREEZE_20261015.md). 선택용 통계.
- [x] T18: 280 fit·20외부단위·감사·공개 검토 PASS. [보고](../reports/LGBM_NESTED_TUNING_20261015.md). 두 튜닝정책 기준 미충족.
- [x] T19: 고정 앙상블 분석·감사·공개 검토 PASS. [보고](../reports/FIXED_ENSEMBLE_20261015.md). 두 기준선 동시 개선 미충족, 커밋 전 준비 산술 절차 예외 보존.
- [x] T20: 20/20·감사·공개 검토 PASS. [보고](../reports/LGBM_GRADE3_WEIGHT_20261015.md). F1 +0.013669이나 반복3→1 902→903으로 기준 미충족.
- [x] T21: 40/40·감사·공개 검토 PASS. [보고](../reports/FEATURE_INTERACTIONS_20261015.md). 곱 Δ−0.000954/구조조합 +0.000384, 둘 다 기준 미충족.
- [x] 컨트롤 타워: T26·T27 완료 근거·공개 보고 검토 및 공통 상태 갱신. 원격 반영·CI는 해당 최종 커밋과 Actions를 근거로 확인한다.
- [x] A1 종합(2026-10-16): T15~T27 감사 집계 JSON만 사용, 신규 학습0. 독립 재계산 검토 PASS_WITH_NOTES 지적 반영. [보고](../reports/EXPERIMENT_SYNTHESIS_20261016.md).
- [x] 평가 기준 변경(2026-10-16 사용자 결정): 층화 5-Fold를 주 평가, geo2/geo1 Group은 심화 분석. A2 [층화 종합](../reports/STRATIFIED_SUMMARY_20261016.md) 신규 학습0·독립 검토 PASS.
- [x] T30(2026-10-16, 주 평가): 41/41·재시도0·독립 감사 PASS_T30·공개 보고 검토 반영. [보고](../reports/RF_FACTORS_STRAT_20261016.md). TE·ROS 제거와 원핫 교체는 모두 악화, 연령 C 정책은 무시할 수준.
- [x] T32(2026-10-16, 주 평가): 118/118·독립 감사 PASS_T32·공개 보고 검토 반영. [보고](../reports/SAMPLING_TARGETS_20261016.md).
- [x] T33(2026-10-16, 주 평가): 20/20·독립 감사 PASS_T33·공개 보고 검토 반영. [보고](../reports/EY_PIPELINE_BENCHMARK_20261016.md). EY 파이프라인 층화 CV 0.707285.
- [x] T34(2026-10-16, 주 평가): 연구30+대조2·독립 감사 PASS_T34·공개 보고 검토 반영. [보고](../reports/DERIVED_BUNDLE_20261016.md). 6개 비교 모두 사전 기준 미충족.
- [x] T31(2026-10-16, 주 평가): 55/55·독립 감사 PASS_T31·공개 보고 검토 반영. [보고](../reports/GEO_CODE_ENCODING_20261016.md). 지역 코드 수치 표현은 DT 외 모든 모델에서 원핫보다 낮음.
- [ ] T29 새 최종 평가 설계: [설계 초안](T29_FINAL_EVALUATION_DESIGN_DRAFT.md)만 작성. 승인 항목 확정 전 manifest 생성·학습·홀드아웃 접근 금지. 공식 개인 최종 모델 T0 LR C=1 유지.
- [ ] T28 3→1 비용 결정 규칙: [설계 초안](T28_GRADE3_SAFETY_PLAN_DRAFT.md)만 작성·독립 검토 반영. 승인 항목(§11) 확정·구현·봉인·실행은 별도 사용자 승인 필요.
- [x] 선택적 후속 정리: T16 임시 실행 폴더는 2026-10-16 안전 확인(추적 변경0, `e7ac78d`는 `peter` 조상) 뒤 사용자 권한 허용으로 제거했다.

각 사전등록 계획과 아래 완료 이력의 브랜치 이름은 당시 출처다. 현재 명령·중단 조건·완료 판정은 [연구 현황](RESEARCH_STATUS.md)의 후속 절차를 따른다. 봉인된 계획의 `prepare` 블록을 다시 실행하지 않는다.

아래는 2026-10-13 개인 분석 완료 기록이다. 후속 연구로 기준선·최종 모델을 자동 변경하거나 홀드아웃을 재평가하지 않는다.

이 브랜치는 Peter 개인 분석 완결본이다. [개인 실행 안내](../README_PERSONAL.md), [최종 보고서](../reports/PERSONAL_FINAL_REPORT_20261013.md), [선택 동결](PERSONAL_MODEL_SELECTION_20261013.md)을 우선한다. 팀 공통 main과 팀원 수행 완료를 대신하지 않는다.

| 단계 | 개인 상태 | 근거 |
|---|---|---|
| 원본·전처리·분할 | 보존 및 검증 완료 | v1.0.1, 38개 입력, 개발/Manifest SHA 불변 |
| 개발 EDA | 55개 집계와 재현 코드 완료 | eda_summary.csv, audit_eda_oof.py |
| 필수 네 모델·가중치·RUS | 비교 완료 | 기존 12개 설정·60 Fold 기록 보존 |
| 제한 튜닝 | 5개 후보·25 Fold 완료 | configs/tuning, dashboard_models.json |
| 일반 SMOTENC | 안전성 파일럿 불채택 | 지역 조합·특수 연령·원본 정수 계약 문제; 전체 CV 실행 아님 |
| 제약형 SMOTE·복제 대조 | 2개 후보·10 Fold 완료 | personal_sampling_summary.csv, 개인 사전 정책 |
| 기존 LR 근거 복구 | 5 Fold 재현 완료 | 이전 점수/혼동행렬 일치, 새로운 OOF; 원래 파일 복구 아님 |
| 개인 최종 후보 | 무가중치 LR C=1 | 19개 고유 설정의 개발 CV 최고 평균 0.683677 |
| 개인 홀드아웃 | 동결 후 1회 평가 완료 | Macro F1 0.686771, Accuracy 74.24%, 재튜닝 없음 |
| Streamlit·예측 CLI | 구현 완료 | 집계 전용 5개 화면, 해시 검사 모델 추론 |
| 실제 데이터 로컬 검사 | 110개 중 103 통과·UI 7 skip | UI는 별도 원격 CI로 검사, 결과는 해당 commit Actions에서 확인 |
| 과거 원래 OOF 8개 | 미확보 | 새 재현/후속 12개 OOF 검증과 구분 |

전체 고유 비교는 19개 설정·95 Fold 기록이며, 근거 복구 5회까지 실제 CV fit은 100회다. 별도로 최종 개발 전체 fit 1회가 있다. 합성 단위 테스트는 학습 실험 수에 합산하지 않는다.

## 팀 문제 정의 반영 (2026-10-15)

`docs/align-problem-definition-20261013`의 목적·오류 해석 합의를 개인 연구에도 반영합니다. 2026-10-13 동의 범위는 2015년 관측 피해 분류, 주요 특성의 연관성, 클래스 불균형 및 심한 피해 과소예측 검토입니다. 공통 분할·오류 비용과 허용 기준·최종 모델·팀 역할은 별도 합의 사항이며 Notion 반영 여부도 별도입니다.

3→1·3→2 건수와 비율 및 3등급 Recall을 함께 제시하고, 오류 비율의 분모는 실제 3등급 건수로 명시합니다. Fold 평균과 pooled OOF는 구분합니다. 미래 지진·다른 지역의 일반화나 인과관계를 검증한 것으로 표현하지 않습니다. 점검·보강 우선순위에 대한 적용에는 지진 강도·지반 조건 등을 고려한 추가 검증이 필요합니다.

해당 브랜치의 팀 공통 영역 정리용 파일 삭제는 `peter`에 적용하지 않습니다. 개인 연구 규칙·실행기·설정·기존 보고서·테스트와 연구 이력을 보존합니다.

## 남은 팀·외부 범위

main 통합, 팀 최종 선정과 평가 정책, 팀원별 전체 과정 수행, Notion/LMS 제출 및 외부 서비스 배포는 별도다. 자체 제약형 SMOTE를 stock SMOTENC 전체 실행이라고 표시하지 않는다. 지진 강도·지반 조건을 포함한 새 지역/미래 지진의 일반화는 검증하지 않았다.

개인 평가로 공통 홀드아웃 결과는 이미 노출됐다. 이를 보면서 다른 팀 후보를 반복 선택한 뒤 같은 자료를 새로운 독립 테스트라고 부르지 않는다. 이미 완료된 개인 모델은 홀드아웃 결과에 따라 재튜닝하지 않는다.

이전 진행 문서는 [원래 계획 보관본](PLAN_BEFORE_PERSONAL_COMPLETION.md)에 보존했다. 팀 README 초안은 저장소 루트의 README_TEAM_DRAFT_ARCHIVE.md에 바이트 그대로 보존했다. 이 완료 상태 변경은 main에는 적용하지 않았다.

## T7 지역 계층적 축소 — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T7_geo_hier_shrink_v1`, 브랜치 `peter-exp/geo-hier-shrink-v1`. [사전 등록 계획](T7_GEO_HIER_SHRINK_PLAN.md).
- C의 범주형 OHE를 geo1/geo2/geo3/text 블록으로 나누고 geo3(또는 geo2+geo3) 블록 가중치 s∈{0.5,0.25,0.1}로 축소한 6개 변형 × stratified/geo2/geo1 × 5 Fold = 신규 fit 90회 예정. C 재학습 없음, 홀드아웃 미사용.
- 상태: 90/90 학습 완료, 독립 감사 PASS. 사전 등록 규칙 충족 변형 0/6, C 기준선 유지. [결과 보고서](../reports/GEO_HIER_SHRINK_20261014.md).

## T8 LightGBM 범주형 표현 — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T8_lgbm_native_cat_v1`, 브랜치 `peter-exp/lgbm-native-cat-v1`. [사전 등록 계획](T8_LGBM_NATIVE_CAT_PLAN.md).
- T4 `LGBM_C_v1`(one-hot, O)과 같은 파라미터·C 연령 정책에서 범주형 11열만 학습 Fold 어휘의 native categorical(N)로 바꾼 단일 요인 비교. stratified/geo2/geo1 × 5 Fold, 신규 fit N 15 + O 10 = 25회. O stratified는 T4 고정 OOF 재사용, 홀드아웃 미사용.
- 상태: 25/25 학습 완료, 독립 감사 PASS. 주 판정(N>O 표현 효과) 충족, 보조 판정(기준선 도전) 미충족(geo2 하락). LR-C 기준선 유지. [결과 보고서](../reports/LGBM_NATIVE_CAT_20261014.md).

## T9 LightGBM 부분 native categorical — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T9_lgbm_partial_native_v1`, 브랜치 `peter-exp/lgbm-partial-native-v1`(기준 `b54b129`). [사전 등록 계획](T9_LGBM_PARTIAL_NATIVE_PLAN.md).
- T8 O와 같은 조건에서 geo 3열은 one-hot을 유지하고 텍스트 범주 8열만 학습 Fold 어휘 native categorical로 바꾼 P(CSR 유지). stratified/geo2/geo1 × 5 Fold = 신규 fit 15회. O·N·LR-C는 기존 OOF 재사용, 홀드아웃 미사용.
- 상태: 15/15 학습 완료, 독립 감사 PASS. 결합 후보 미충족, N−O 이득 유지 비율 0.4%(이득은 지역 범주에서 옴). LR-C 기준선 유지. [결과 보고서](../reports/LGBM_PARTIAL_NATIVE_20261014.md).
## T10 지역 계층적 Target Encoding — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T10_geo_hier_te_v1`, 브랜치 `peter-exp/geo-hier-te-v1`. [사전 등록 계획](T10_GEO_HIER_TE_PLAN.md).
- 단일 요인: 지역 3열의 one-hot 블록을 학습 Fold 안에서만 만든 계층적 TE 9개 특성(te1/te3/cnt × geo1/2/3, m=20, 내부 5-Fold OOF)으로 교체. LR_TE(vs LR-C)와 LGBM_TE(vs one-hot LGBM O) × stratified/geo2/geo1 × 5 Fold = 신규 fit 30회 예정. 기준선 재학습 없음, 재표집·튜닝 없음, 홀드아웃 미사용.
- 상태: 30/30 학습 완료, 독립 감사 PASS. 사전 등록 규칙 4개 모두 미충족(LR-C 기준선 유지). Group CV는 학습·검증 TE 분포 차이로 예측이 쏠려 해석 제한. [결과 보고서](../reports/GEO_HIER_TE_20261014.md).

## T12 지역 ID 없는 구조 파생 특성 — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T12_struct_features_v1`, 브랜치 `peter-exp/struct-features-v1`. [사전 등록 계획](T12_STRUCT_FEATURES_PLAN.md).
- 단일 요인: 지역·연령을 쓰지 않는 구조 파생 특성(번들 S 4개, D 3개, SD 7개)을 각 계열의 기존 행렬 뒤에 덧붙임. LR(vs LR-C)·LGBM(vs one-hot O) × 3 번들 × stratified/geo2/geo1 × 5 Fold = 신규 fit 90회 예정. 기준선 재학습 없음, 홀드아웃 미사용.
- 상태: 90/90 학습 완료, 독립 감사 PASS. 공동 개선 후보 0/6(최대 Δ stratified +0.0009, geo2 +0.0011 < 기준 +0.003). 기준선 유지. [결과 보고서](../reports/STRUCT_FEATURES_20261014.md).
## T11 그룹 인지 내부 Fold 지역 Target Encoding — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T11_geo_te_group_inner_v1`, 브랜치 `peter-exp/geo-te-group-inner-v1`(기준 `7956901`). [사전 등록 계획](T11_GEO_TE_GROUP_INNER_PLAN.md).
- 단일 요인: T10에서 학습 행을 인코딩하는 내부 OOF 분할기를 바깥 분할을 모사하는 `StratifiedGroupKFold`(geo2: (geo1,geo2) 그룹, geo1: geo1 그룹)로 교체. LR_TE_G/LGBM_TE_G × geo2/geo1 × 5 Fold = 신규 fit 20회 예정. stratified는 내부 분할이 T10과 같아 T10 OOF 재사용. 기준선 재학습 없음, 홀드아웃 미사용.
- 상태: 20/20 학습 완료, 독립 감사 PASS. 분포 복구 LR·LGBM 충족, LGBM 지역 일반화 후보 충족(경계: geo2 +0.01013, 4/5), LR-C 도전 미충족. 확정 아님(별도 확인 필요), LR-C 기준선 유지. [결과 보고서](../reports/GEO_TE_GROUP_INNER_20261014.md).

## T13 LGBM_TE_G geo2 확인 (새 seed의 Group 분할) — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T13_geo2_te_confirm_v1`, 브랜치 `peter-exp/geo2-te-confirm-v1`(기준 `30c3949`). [사전 등록 계획](T13_GEO2_TE_CONFIRM_PLAN.md).
- T11의 경계 충족(단일 T5 geo2 분할, seed 42)을 새 geo2 Group 분할 seed 101/202/303에서 확인. LR_C·LGBM_O·LGBM_TE_G × 3 seed × 5 Fold = 45 fit(최대 45). 규칙 C1·C2는 계획 문서에 학습 전 고정.
- 상태: 45/45 학습 완료, 독립 감사 PASS. C1(one-hot LGBM 대비) 미충족(+0.0086, 11/15), C2(LR-C 대비) 충족(+0.0243, 14/15) → 확인 실패, LR-C 기준선 유지. [결과 보고서](../reports/GEO2_TE_CONFIRM_20261014.md).

## T14 TE·ROS RandomForest 탐색 결과 로컬 재현 — 2026-10-15 (완료, 독립 감사 PASS)

- 실험 ID `T14_TE_ROS_RF_LOCAL_REPRO`, 브랜치 `peter-exp/te-ros-rf-local-repro-v1`(기준 `af261c7`). [사전 등록 계획](T14_TE_ROS_RF_REPRO_PLAN.md).
- `origin/peter-exp/te-ros-rf-exploratory-v1`(`0ecd351`)의 미감사 탐색 결과(RF + geo1/2/3 OOF TE + 지역 빈도 + 학습 Fold 한정 ROS, Macro F1 0.700427)를 원본 코드 보존본(`experiments/te_ros_rf_exploratory/`, SHA 고정)으로 로컬 재현한다. **재현이며 요인 분리(TE/ROS/RF/age995)는 후속 실험이다.**
- Stage A: 고정 stratified 5 Fold, 기록된 Fold별 혼동행렬과 비교(EXACT / WITHIN_TOLERANCE / FAIL, 허용 오차 사전 고정). Stage B(Stage A 통과 시만): geo2 Group 분할 seed 42/101/202/303 20 Fold, 내부 TE Fold는 T11 그룹 인지 방식, 기존 LR-C·LGBM-O·LGBM_TE_G OOF 재사용(비교군 재학습 없음). 신규 RF 학습 25회 예정.
- 상태: **완료, 독립 감사 PASS** (시도 2 `run_v2`, 수정 1 `813fab3`). [결과 보고](../reports/TE_ROS_RF_REPRO_20261015.md). Stage A는 EXACT로, 0.7004를 혼동행렬 단위까지 재현했다. Stage B는 NEITHER로, geo2 20 Fold에서 LR-C −0.0088(8/20 승), LGBM-O −0.0250(2/20 승)이다. 기준선·최종 모델·홀드아웃은 변경하지 않았다.
## T15 3등급 결정 규칙(로짓 오프셋) — 2026-10-15 (완료, 독립 감사 PASS)

- 실험 ID `T15_GRADE3_DECISION_RULE`, 브랜치 `peter-exp/grade3-decision-rule-v1`(기준 `af261c7`). [사전 등록 계획](T15_GRADE3_DECISION_RULE_PLAN.md).
- 단일 요인: 3등급 확률에 `exp(b)`를 곱하는 결정 규칙(b는 0~2.0, 0.1 간격 21개 중 1개). LR_C·LGBM_O·LGBM_TE_G에 동일 적용, 각 팔의 보정 전 argmax와 비교. 누수 방지를 위해 b는 바깥 학습 행 안의 내부(Group 인지) 5-Fold OOF로만 선택하고, 바깥 모델은 재학습하지 않고 저장 OOF를 읽기 전용 재사용.
- 분할: geo2 seed 42/101/202/303(주, 20 Fold), stratified seed 42(낙관적, 보조), geo1 seed 42(진단). 신규 fit은 내부 Fold 450회(최대 450), 홀드아웃 미사용.
- 상태: **완료, 독립 감사 PASS** (사전 등록 `7a8796b`, 내부 fit 450회). [결과 보고](../reports/GRADE3_DECISION_RULE_20261015.md). geo2 기준 LGBM-O(+0.0138, 20/20)와 LGBM_TE_G(+0.0046, 17/20)에서는 보정 규칙이 효과 기준을 충족했고, LR-C(+0.0022)는 P1 미달이다. 공통 규칙은 성립하지 않는다. 이는 규칙 효과 판정이며 최종 모델 후보 판정이 아니다. 기준선·최종 모델·홀드아웃은 변경하지 않았다.
