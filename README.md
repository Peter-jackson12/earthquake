# Earthquake — Peter 개인 분석 완결본

**최종 모델: v1.0.1 전처리 + 무가중치 Logistic Regression C=1.**
개발 CV Macro F1 **0.683677**, 개인 홀드아웃 Macro F1 **0.686771**, Accuracy **74.24%**.
19개 고유 설정 비교 후 후보를 동결하고 52,121건을 한 번 평가했다. 개인 결과이며 팀 최종 선정/제출 완료가 아니다.

## 후속 연구 인계·중복 실행 방지

**[완료된 실험·진행 중인 연구 트랙과 중복 실행 방지 체크리스트](docs/NEXT_EXPERIMENTS_20261013.md)**를 먼저 확인한다. 기존 `peter` 개인 최종 모델과, 별도 `peter-exp/age995-outlier-treatment`에서 완료한 연령995 처리·오류 진단·G1~G5 실험은 **서로 다른 연구 단계**다. XGBoost·LightGBM 비교는 다른 브랜치에서 진행 중이므로 최신 상태를 확인한 뒤 이어간다. 이미 사용된 개인 홀드아웃은 재평가하지 않는다.

## 연구 참고 문서 (로컬 에이전트·팀 공유)

- [실제로 구현·실험한 파생변수 인벤토리](docs/FEATURE_ENGINEERING_INVENTORY_20261014.md): age_is_995, 연령 8구간과 지역·기초·연령 교차항의 정확한 정의·사용 여부. 원핫 열, 분할키, 샘플링과 구분.
- [외부 논문·GitHub 벤치마크 비교](docs/EXTERNAL_BENCHMARK_COMPARISON_20261014.md): 원문 링크, 저자 보고 F1, Macro/Micro/Accuracy·평가 분할 차이, 독립 검증 한계.

이 문서는 **참고/인계용**이며 신규 학습·최종 모델 교체 지시가 아니다. T4/T5/T6 최신 상태는 각 실험 브랜치의 완료 보고·감사 기록을 우선한다. 특히 T6 R12/R1/R0의 성능은 기존 비공개 증거 복구 전까지 확정하지 않는다.

## 개인 연구 실험 목록 (2026-10-15, `peter`로 통합)

`peter-exp/*` 실험 브랜치의 이력은 이 브랜치에 병합했다. 각 브랜치의 원래 끝 커밋은 `archive/peter-exp/<브랜치 이름>` 태그로 보존한다. 보고서에 적힌 커밋 SHA는 그대로 유효하다. 개인 최종 모델(LR C=1)은 위 결과 그대로이고, 아래 실험은 모두 **개발 CV 연구**다(홀드아웃 재사용 없음, 기준선 LR-C 유지). 진행 상태는 [docs/PLAN.md](docs/PLAN.md)와 [연구 현황](docs/RESEARCH_STATUS.md)을 따른다.

| 단계 | 주제 | 보고서 |
|---|---|---|
| T0 | 기존 개인 모델(위 최종 모델) | [개인 최종 보고](reports/PERSONAL_FINAL_REPORT_20261013.md) |
| T1 / T2 | `age=995` 전처리(A/B/C, D~F) / 오류 진단 | [실험](reports/AGE995_EXPERIMENT_20261013.md), [후속](reports/AGE995_FOLLOWUP_20261013.md), [진단](reports/AGE995_DIAGNOSTICS_20261014.md) |
| T3 | 특성 조합 G1~G5 | [보고](reports/AGE995_INTERACTIONS_20261014.md) |
| T4 | 모델 계열 비교(XGBoost·LightGBM vs LR) | [보고](reports/MODEL_FAMILY_COMPARISON_20261014.md), [재현 검토](reports/MODEL_FAMILY_REPRODUCTION_REVIEW_20261014.md) |
| T5 | 지역 Group CV | [보고](reports/GEO_GROUP_CV_20261014.md) |
| T6 / T6R / T6S | 지역 특성 제거 ablation·로컬 재현·C 보완 감사 | [회수](reports/GEO_ABLATION_RETRIEVAL_20261014.md), [복구](reports/GEO_ABLATION_RECOVERY_20261014.md), [T6R](reports/GEO_ABLATION_LOCAL_REPRO_20261014.md), [T6S](reports/GEO_ABLATION_C_GROUP_SUPPLEMENT_20261014.md) |
| T7 | 지역 계층 축소 | [보고](reports/GEO_HIER_SHRINK_20261014.md) |
| T8 / T9 | LightGBM native categorical / 부분 native | [T8](reports/LGBM_NATIVE_CAT_20261014.md), [T9](reports/LGBM_PARTIAL_NATIVE_20261014.md) |
| T10 / T11 | 계층적 지역 TE / 그룹 인지 내부 Fold TE | [T10](reports/GEO_HIER_TE_20261014.md), [T11](reports/GEO_TE_GROUP_INNER_20261014.md) |
| T12 | 구조 파생 특성 | [보고](reports/STRUCT_FEATURES_20261014.md) |
| T13 | LGBM_TE_G geo2 확인(새 분할) | [보고](reports/GEO2_TE_CONFIRM_20261014.md) |
| 탐색 / T14 | TE·ROS RandomForest 탐색(0.7004) / 로컬 재현 | [탐색](reports/TE_ROS_RF_EXPLORATORY_20261014.md), [재현](reports/TE_ROS_RF_REPRO_20261015.md) |
| T15 | 3등급 결정 규칙(로짓 오프셋) | [보고](reports/GRADE3_DECISION_RULE_20261015.md) |

## 결과 보기 — 데이터나 모델 파일 없이 실행

```bash
python -m pip install -r requirements-dashboard.txt
python -m streamlit run app.py
```

실험 비교 / 개발 EDA / 오류 분석 / 개인 최종 평가 / 검증·진행 상태를 제공한다.
`reports/PERSONAL_FINAL_REPORT_20261013.md`에 처리 정책, 실패한 개선, 최종 등급별 결과와 한계를 기록했다.

## 비공개 예측 모델 사용

전달한 private 증거의 `outputs/personal_final_v1/model.joblib`를 프로젝트의 동일 위치에 둔다. 공개 Git에는 모델을 넣지 않는다.
입력 CSV는 `building_id` + 원본 38개 입력 순서이며 `damage_grade`를 포함하지 않는다.

```bash
python scripts/predict_personal.py --input local_buildings.csv --output outputs/local_predictions.csv
```

CLI는 모델 SHA를 `reports/personal_final_holdout.json`과 대조한다. 출처가 불명확한 joblib 파일은 로딩하지 않는다. 이 교육용 모델을 실무 안전 판정에 사용하지 않는다.

## 실험 재현

전처리 v1.0.1 패키지의 `data/`를 비공개로 설치한다. `configs/data_contract.json`의 개발/분할 SHA가 맞아야 한다.
기존 outputs가 없는 깨끗한 작업 복사본에서 아래 CV를 재현한다. 기존 결과 디렉터리는 덮어쓰지 않는다.

```bash
python scripts/run_personal_sampling.py --mode none --output outputs/personal_none_v1
python scripts/run_personal_sampling.py --mode random_over --output outputs/personal_random_over_v1
python scripts/run_personal_sampling.py --mode context_smote --output outputs/personal_context_smote_v1
python -m unittest discover -s tests -v
```

`none`은 기준선 설정의 근거 복구 재실행으로 새 고유 후보가 아니다. 기존 5개 튜닝 후보는 `configs/tuning/`의 설정을 `baseline.py --config ... --models lr` 또는 `--models rf`로 재현할 수 있다. 기록된 이전 성능을 새 실행 수치로 덮어쓰지 않는다.

`personal_reference_snapshot.json`은 선택 전 17개 설정 집계, `personal_selection.json`은 후보 동결 기록이다.
`scripts/finalize_personal.py freeze`와 `evaluate --selection-sha ...`는 최초 선택/평가 절차를 보존한 코드다. 완료된 개인 분석에서 다시 실행해 점수를 고르지 않는다. 이미 동결 파일 또는 최종 출력이 있으면 거부한다.

## 전처리·불균형 처리 결정

일반 SMOTENC는 잘못된 지역 조합 및 특수 연령 보간 문제로 불채택했다. 동일한 범주·층수·가구수·연령 상태 안에서만 수치를 보간하는 자체 제약형 SMOTE+복제 대안은 실제 5-Fold로 비교했다. 표준 SMOTENC와 같은 구현이 아니다. 새 방식의 CV Macro F1은 0.663206, 단순 복제 대조군은 0.663369로 기존 LR보다 낮아 최종 모델에는 쓰지 않았다.

## 경계

원본 CSV, 행별 라벨/예측, 모델과 자격증명은 공개 Git 제외다. 이전 8개 원래 OOF는 미확보이며 새 재현 파일과 구분한다.
개인 평가 후 공통 홀드아웃은 이미 노출됐다. 이 결과를 보며 팀 모델을 다시 선택한 뒤 같은 점수를 독립 테스트라고 부르지 않는다. main 통합·팀원별 수행·Notion/LMS 제출·외부 서비스 배포는 별도다.
