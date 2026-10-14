# 개인 분석 완료 상태 — 2026-10-13

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

## T10 지역 계층적 Target Encoding — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T10_geo_hier_te_v1`, 브랜치 `peter-exp/geo-hier-te-v1`. [사전 등록 계획](T10_GEO_HIER_TE_PLAN.md).
- 단일 요인: 지역 3열의 one-hot 블록을 학습 Fold 안에서만 만든 계층적 TE 9개 특성(te1/te3/cnt × geo1/2/3, m=20, 내부 5-Fold OOF)으로 교체. LR_TE(vs LR-C)와 LGBM_TE(vs one-hot LGBM O) × stratified/geo2/geo1 × 5 Fold = 신규 fit 30회 예정. 기준선 재학습 없음, 재표집·튜닝 없음, 홀드아웃 미사용.
- 상태: 30/30 학습 완료, 독립 감사 PASS. 사전 등록 규칙 4개 모두 미충족(LR-C 기준선 유지). Group CV는 학습·검증 TE 분포 차이로 예측이 쏠려 해석 제한. [결과 보고서](../reports/GEO_HIER_TE_20261014.md).

## T11 그룹 인지 내부 Fold 지역 Target Encoding — 2026-10-14 (완료, 독립 감사 PASS)

- 실험 ID `T11_geo_te_group_inner_v1`, 브랜치 `peter-exp/geo-te-group-inner-v1`(기준 `7956901`). [사전 등록 계획](T11_GEO_TE_GROUP_INNER_PLAN.md).
- 단일 요인: T10에서 학습 행을 인코딩하는 내부 OOF 분할기를 바깥 분할을 모사하는 `StratifiedGroupKFold`(geo2: (geo1,geo2) 그룹, geo1: geo1 그룹)로 교체. LR_TE_G/LGBM_TE_G × geo2/geo1 × 5 Fold = 신규 fit 20회 예정. stratified는 내부 분할이 T10과 같아 T10 OOF 재사용. 기준선 재학습 없음, 홀드아웃 미사용.
- 상태: 20/20 학습 완료, 독립 감사 PASS. 분포 복구 LR·LGBM 충족, LGBM 지역 일반화 후보 충족(경계: geo2 +0.01013, 4/5), LR-C 도전 미충족. 확정 아님(별도 확인 필요), LR-C 기준선 유지. [결과 보고서](../reports/GEO_TE_GROUP_INNER_20261014.md).
