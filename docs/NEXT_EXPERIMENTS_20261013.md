# 후속 실험 상태와 아직 남은 비교 — 개인 peter

2026-10-13 후속 실행 반영. [현재 결과](../reports/FOLLOWUP_REPORT_20261013.md)와 [재현 명령](FOLLOWUP_RUNBOOK_20261013.md)을 먼저 확인한다.
개인 선행 작업이며 팀 합의·팀원별 전체 과정 수행을 대신하지 않는다. 기존 80/20, seed 42, 고정 5-Fold는 팀 최종 합의 전 잠정 기준이다.

## 2026-10-14 별도 개인 연구 인계 — 중복 실행 방지 (최신 확인 스냅샷)

> **새 세션 우선 확인:** 이 `peter` 브랜치의 기존 개인 최종 모델과, 이후 `peter-exp/age995-outlier-treatment`에서 수행한 *새 개발 CV 연구*는 별개다. 아래 표는 **2026-10-14 KST에 확인한 근거 커밋** 기준이며, 최신 진행 상태는 작업 전 각 브랜치 HEAD와 결과 보고서를 다시 확인한다. 최종 모델·앱·팀 `main`은 이번 연구 결과로 교체하지 않았다.

| 연구 트랙 | 재실행 판단용 현재 상태 | 결론·검증 근거 |
|---|---|---|
| T0 — `peter` 기존 개인 완결 | **완료; 최종 홀드아웃 재평가 금지** | [개인 최종 보고](../reports/PERSONAL_FINAL_REPORT_20261013.md), [개인 완료 상태](PLAN.md). 무가중치 LR `C=1`의 기존 개발 CV 0.683677, **이미 1회 사용한** 개인 홀드아웃 Macro F1 0.686771. 여기서 `C=1`은 LR 규제 하이퍼파라미터이며, 다음 T1의 *전처리 정책 C*와 다름. |
| T1 — `age=995` LR 처리 | **9개 정책 A/B/C/D/E1/E2/E3/F1/F2 비교 및 감사 완료; 동일 조건 중복 학습 불필요** | [A/B/C](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_EXPERIMENT_20261013.md), [D/E/F](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_FOLLOWUP_20261013.md). 정책 C(학습 Fold 비995 연령 중앙값 15로 995만 대체 + 원본995 표시) CV Macro F1 **0.688687**; E2(상한90 + 표시) **0.688664**. 삭제 정책 D/F1/F2는 995 건물 1,125개를 전부 3등급으로 분류하는 문제가 발생. |
| T2 — C/E2 및 비995 오류 분석 | **재학습 없이 OOF 감사 완료; 동일 오류 집계 반복 불필요** | [독립 진단](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_DIAGNOSTICS_20261014.md). C의 전체 3→1 오류 **452건 중 447건은 비995**, 지역 코드 26/20/13에서 292건(64.60%); 지역·연령·구조는 *관찰된 오류 분포*이지 인과 원인 확정 아님. |
| T3 — C 기반 G1~G5 추가 특성 | **5설정 × 5Fold = 신규 25회 실학습·OOF 감사 완료** | [고정 계획](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/experiments/age995/interactions/PLAN.md), [실제 결과](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/reports/AGE995_INTERACTIONS_20261014.md). 최고 점수 G2(연령 구간 + 지역×기초구조) CV Macro F1 **0.690993**. 단, 전체 3→1 오류 **C 452→G2 478**, 원본995 그룹 성능 저하. **사전 보수적 선택 조건을 통과하지 못해 LR 후속 연구 기준 C 유지; G2는 점수 개선 후보만 보존.** |
| T4 — XGBoost/LightGBM 계열 비교 | **다른 세션 진행 중; 완료 수치 아직 이 인계에서 검증하지 않음** | [별도 작업 브랜치](https://github.com/Peter-jackson12/earthquake/tree/peter-exp/model-family-comparison-v1)의 사전 계획 커밋 `e0dd14630b9d20aa116fee57aa6555102c15c01c` 확인. 해당 세션의 최신 코드·OOF·보고서를 확인하기 전 학습을 중복 실행하거나 완료로 표시하지 말 것. [시작 인계](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/docs/HANDOFF_MODEL_FAMILY_COMPARISON_20261014.md). |

**반드시 구분:** T1~T3 점수는 개발 **208,480행·기존 Stratified 5Fold·seed42**의 탐색적 CV다. `peter`에서 **이미 사용된 홀드아웃 52,121행을 다시 열거나 모델 선택·튜닝·검증에 사용하지 않는다.** 새 독립 최종성능, 다른 지진에서의 일반화, 실제 안전성 입증으로 쓰지 않는다. 원시 데이터·행별 OOF·모델 바이너리는 공개 Git에 올리지 않는다.

**새 세션 최소 절차:** (1) `peter` 및 각 연구 브랜치의 최신 HEAD 확인 → (2) [실험 브랜치의 연구 현황](https://github.com/Peter-jackson12/earthquake/blob/a4fc5c82a1907d8fc975f0971deaed8d26a4461f/docs/RESEARCH_STATUS.md)과 위 **최신 T3 보고서** 대조(해당 상태 문서에는 예전 'T3 진행 중' 스냅샷이 남아 있을 수 있음) → (3) 동일 데이터 SHA·Fold·설정·OOF/감사 근거 확인 → (4) **완료된 실험은 재학습하지 않고 기존 집계·감사 결과 재사용**, 새로운 가설만 새 ID·브랜치·출력 경로에서 실행.

**아래 「다음 1~3」은 2026-10-13 작성 당시의 계획이므로 최신 할 일 목록이 아니다.** `peter`에서 제한 튜닝·제약형 SMOTE/복제 비교·개인 최종 홀드아웃 평가까지 이미 완료된 범위는 [개인 완료 상태](PLAN.md)와 [개인 최종 보고](../reports/PERSONAL_FINAL_REPORT_20261013.md)를 우선한다. 팀 모델 합의·제출·외부 배포는 별도이며, T4 결과는 해당 세션의 최신 근거로 갱신해야 한다.

## 중복 실행하지 않을 완료 항목

| 항목 | 상태 | 근거 |
|---|---|---|
| 전처리 v1.0.1·개발 데이터 EDA | 검증 및 재현 코드 완료 | scripts/audit_eda_oof.py, reports/eda_summary.csv |
| Dummy+필수 4개 모델 기준선, RF/GBC max_features, LR balanced | 이전 8개 설정·40 Fold 보존 | BASELINE_REPORT.md 및 기존 summary CSV |
| DT/RF balanced, GBC 학습 Fold sample_weight | 신규 3개 설정·15 Fold 완료 | configs/class_weights.json, followup_model_summary.csv |
| LR Random Under Sampling | 신규 1개 설정·5 Fold 완료 | configs/lr_rus.json, followup_model_summary.csv |
| 신규 4개 설정 OOF·지역 오류 | 행별 OOF에서 검증 완료 | followup_verification.json |
| 과거 8개 설정의 행별 OOF 재검증 | 미완료 | 원래 OOF 파일 부재. 새 원본 CSV로 복원된 것으로 간주하지 않음 |

기존 기준선을 검토하려고 40회 학습을 반복하지 않는다. 새 결과가 과거 지역 26의 LR 기본·balanced 오류 177/632건을 소급 검증하지는 않는다.

## 다음 1 — 제한 CV 튜닝

기본 우선 후보는 무가중치 LR이며 가중치 RF도 비교 후보로 유지한다. 최종 모델 선정은 아니다.

- LR의 새 후보: `C={0.1,10}`. 이미 실행한 무가중치 `C=1`은 같은 provenance의 기준점으로 재사용한다.
- 기존 무가중치 RF 기준에서 새 조합은 `(max_features,min_samples_leaf)={(0.1,1),(0.2,1),(0.2,2)}`다. `(0.1,2)`는 이미 실행했다.
- 가중치 RF를 튜닝할 경우 먼저 `class_weight=balanced`를 고정하고 별도 실험 ID로 기록한다. 무가중치 RF 결과와 섞거나 두 변경을 하나의 효과로 주장하지 않는다.
- 기존과 동일한 평가 ID·Fold·38개 입력·Macro F1을 유지한다. Fold별 방향, 등급별 P/R/F1, 3→1·3→2, 경고와 학습 비용을 함께 보고한다.
- 예산과 탐색 공간을 실행 전에 정하고 결과를 본 뒤 사후 기준으로 홀드아웃까지 조정하지 않는다.

## 다음 2 — SMOTENC 안전성 파일럿

아직 구현·실행하지 않았다. 원핫 행렬에 일반 SMOTE를 바로 적용하지 않는다.
기존 원본 정수 계약은 `age=10.5`를 거부한다. 따라서 합성 연속값을 기존 스키마 검사기에 그대로 넣는 단순 연결은 사용하지 않는다.

1. 원본 스키마 검증은 유지하고, 합성 샘플 후처리·인코딩을 별도 실험 경로로 설계한다. v1.0.1을 몰래 완화하거나 임의 반올림하지 않는다.
2. 지역 3개·문자 8개·이진 22개의 명목형 지정, 수치 5개의 거리 스케일, age=0/995 의미와 보간 제한을 명시한다.
3. 학습 Fold에서만 이웃/스케일 통계를 계산한다. 생성된 지역 계층 조합, 이진 범위, 특수값과 상충하는 조합을 점검한다.
4. 소규모 파일럿에서 시간·메모리·유효성·클래스 수를 기록하고 전체 Fold 실행 여부를 결정한다. 작은 파일럿을 필수 전체 비교 완료로 세지 않는다.

RUS와 가중치가 완료됐어도 SMOTE 계열 비교의 미실행을 숨기지 않는다.

## 다음 3 — 팀 선택·최종 평가·제출

팀 공통 기준 합의 → 선택한 후보 확정 → 남겨둔 홀드아웃 최종 평가 순서를 지킨다.
Streamlit은 실제 EDA와 기존/후속 집계를 이용해 구축하되 최종 평가 칸은 실행 전까지 비워 두거나 미실행으로 표시한다.
팀 Notion·개인 수행 기록·README·main 제출 통합은 별도 남은 작업이다. peter 전체를 main에 무조건 병합하지 않는다.
지역 Group CV, 변수 묶음 제거, SHAP, 추가 모델은 필수 작업 이후의 선택 심화다.
