# 네팔 지진 피해등급 외부 논문·프로젝트 벤치마크 메모

확인일: **2026-10-14 KST**. 공개 **저자·저장소가 보고한 점수**를 옮긴 문헌 조사이며, 이 저장소에서 동일 코드를 재실행하거나 원본 OOF를 검증한 결과가 아니다. 로컬 에이전트와 연구자가 방법 후보를 **분리된 실험**으로 검토할 수 있게 기록한다.

## 1. 동일 3등급 Richter's Predictor 데이터 관련 수치

| 외부 출처 | 발표·저장소 모델 | 외부 보고 수치 | 지표와 신뢰 범위 | 관련 방법 |
|---|---|---|---|---|
| [Poudyal & Shakya, FAITH (2025)](https://doi.org/10.62411/faith.3048-3719-109) | Random Forest + SMOTE | **Macro F1 0.67**; RandomOverSampling **0.65** | 논문 초록의 저자 수치; 분할·전처리 재현 검증 안 함 | 불균형 처리, IQR/조건 기반 이상치 제거 |
| [Sahil Ali, earthquake-damage-prediction](https://github.com/sahilali09/earthquake-damage-prediction) | XGBoost | **홀드아웃 Macro F1 0.6896**, Accuracy/Micro F1 0.7426 | 저장소 README의 자체 보고치; 우리 홀드아웃과 같은 분할이라 보장 안 됨 | 원본 38개 + 재료 강도/개수/사용처 기반 파생항 6개, 995 중앙값 대체 |
| [kuchbhi-xyz-hub/Earthquake-Damage-Prediction](https://github.com/kuchbhi-xyz-hub/Earthquake-Damage-Prediction) | Native Categorical LightGBM + 의사결정 규칙 조정 | **5-Fold Macro F1 0.7082**, 홀드아웃 Macro F1 **0.7171** | 저장소 README의 자체 보고치; 독립 복제/누수/튜닝 경계 미확인 | 범주 처리·예측 규칙을 함께 개선. 어느 요소가 얼마나 기여했는지는 분리 실험 필요 |
| [Dxzhang·Worley 등 (2021)](https://medium.com/@dxzhang0921/nepal-earthquake-damage-grade-modeling-367b3c2d95e2) 및 [코드](https://github.com/sworley1/AdvML-TermProject) | XGBoost / CatBoost | **Accuracy 72.8% / 72.4%** | **Macro F1 미확인**, 따라서 0.728/0.724를 Macro F1으로 옮기지 말 것 | 995 보존; 지리/구조 특성 비교, PCA 시도 |
| [Atheer (2020)](https://medium.com/swlh/classification-of-nepals-2015-earthquake-damage-levels-with-gradient-boosting-and-smote-6449234ec649) | GradientBoosting + 특성 선택 + SMOTE | 저자 표기 F1 **0.442 → 0.448 → 0.556** | **F1의 평균 방식 미확인**; 글의 Accuracy와 혼동행렬이 정합되지 않아 벤치마크로 직접 사용 금지 | 특성 선택 및 불균형 처리의 별도 비교 아이디어 |

각 출처의 성능은 **동일 분할·동일 입력·동일 평가 방법·동일 전처리라고 확인하지 않았다**. 숫자의 대소 관계로 우리 성능의 우열을 판정하지 않는다.

우리 공개 연구 기준은 `C`의 개발 Stratified 5-Fold **Macro F1 0.688687** (과거 T1); T4의 XGB/LGBM은 같은 기존 원핫 입력으로 LR-C보다 낮았다. T5에서 C가 미관측 1차 지역 Group CV **Macro F1 0.365650**으로 크게 저하됐다. 외부 자료에서 *동일 Group CV 검증이 확인되지 않으면* 지역 일반화 우수성을 주장하지 않는다.

## 2. 유의할 점: 데이터/지표/누수

1. DrivenData 공식 대회 지표는 **Micro F1**이며 단일 정답 다중분류에서 Accuracy와 동일하다. 우리 연구 **Macro F1**과 직접 비교 금지.
2. Atheer의 표시된 혼동행렬에서 직접 계산한 Accuracy와 글에 기재된 Accuracy의 일치 여부에 문제가 있다. F1 0.556을 **Macro F1로 확정하지 않음**.
3. FAITH 논문 수치와 자체 38열 입력의 전처리/행 제거 정책은 독립 재현 전까지 저자 주장으로 다룬다. 논문이 설명한 `age>55`, `floors>5`, `area>20`, `height>10` 제거는 우리 기존 필터·성능 정답이 아니다.
4. Sahil Ali의 앱 README는 `age_is_missing`, `weak_material_count`, `strong_material_count`, `material_quality_score`, `total_secondary_uses`, `vulnerability_index` 여섯 파생항을 열거한다. **이 6개는 외부 프로젝트의 구현이지 우리 Earthquake 개인 코드의 파생변수가 아니다.** 995의 공식 의미는 확인되지 않았다.
5. kuchbhi 프로젝트의 CV와 test 점수는 저자 README 수치다. Native Categorical, 하이퍼파라미터 튜닝, 의사결정 규칙 조정의 효과를 한꺼번에 우리 기존 T4 대비 '범주형 처리의 효과'로 간주하지 않는다. 그 저장소의 52,121행 test와 우리의 내부 홀드아웃은 건수가 같더라도 샘플/분할 동일성은 미확인이다.
6. Zhang 프로젝트의 연령995 역사적 해석은 공식 입증되지 않았다. 일부 코드의 전체 데이터 기반 변환·검증 데이터 재사용 가능성을 확인하지 않고 방법을 복제하지 않는다.
7. 외부 3등급 피해등급 분류와 다른 원본/리코딩/5등급 또는 이진 피해 등급의 논문 결과를 동일 비교표에 혼합하지 않는다. [기존 문헌 메모](PAPER_BENCHMARK_NOTES_KO.md) 및 [정정](PAPER_BENCHMARK_CORRECTIONS_KO.md) 참고.

## 3. 후속 연구에 적용하는 방식 (제안일 뿐 실행 지시 아님)

- **T6 보완 중**: 기존 지역 제거 R12/R1/R0 실행 증거가 없어 효과가 미검증. T6의 원시 OOF·모델·원장을 회수해 감사하기 전, 외부 점수 때문에 새 모델 학습을 시작하지 않는다.
- **T7 후보 A**: Native Categorical을 T4의 고정 모델/분할과 대조하여 표현 효과만 분리. 외부 프로젝트처럼 하이퍼파라미터·임계값을 동시에 바꾸지 않는다.
- **T7 후보 B**: T6에서 지역 제외 효과가 검증될 경우, 그 입력 구성을 트리 모델에도 적용. 적절한 Group CV 트리 기준선을 먼저 확보한다.
- **별도 가설**: Sahil 프로젝트의 재료·건물 사용 기반 파생항은 공식 스키마와 계산식을 확인한 후 별도 연구 ID로 평가. 성능을 봤던 동일 개발 CV의 반복 탐색 편향을 명시한다.
- **평가 일관성**: Fold Macro F1 평균±표준편차와 pooled OOF를 구분; 등급별 Recall, 3→1/3→2, 원본995 집단, 지역 Group CV를 모두 확인. 훈련 Fold에서만 통계 학습. 기존 홀드아웃 재사용·팀 모델 자동 교체 금지.

## 4. 이전에 보관한 추가 참고

[논문 비교·검증 설계 메모](PAPER_BENCHMARK_NOTES_KO.md)에는 Ghimire et al. (Earthquake Spectra; 5등급·더 큰 자료), ACK (2023), Tennessee Tech 포스터 (2021), IOE Graduate Conference (2022), Stanford CS229 Liang (2021), DrivenData 공식 가이드가 링크되어 있다. 이들은 **평가 설계와 발표 구조 참고**이며 우리 3등급 Macro F1의 동일 벤치마크로 옮기지 않는다. [age=995 외부 처리 방식 4건](AGE_995_REFERENCE_NOTES_KO.md)도 함께 읽는다.

## 5. 로컬 에이전트 재현·인계 기준

- 읽기 시작: `docs/EXTERNAL_BENCHMARK_COMPARISON_20261014.md` → `docs/PAPER_BENCHMARK_NOTES_KO.md` → `docs/FEATURE_ENGINEERING_INVENTORY_20261014.md` → 실험별 PLAN/REPORT.
- 출처의 **저자 보고 수치**와 **우리 저장소의 검증된 실측치**를 반드시 분리한다. 앞의 표는 우리 실험 결과가 아니다.
- 기존 T1~T6 작업을 이 문헌 메모만 보고 재실행하지 않는다. 개발 SHA, 봉인 설정, 각 연구의 감사 영수증부터 확인한다.
- 원시 데이터, 개인 홀드아웃, 행별 OOF, joblib, 비공개 ZIP을 공개 저장소에 추가하지 않는다. 외부 노트북이나 논문 전문을 복제하지 않고 링크와 자체 요약만 유지한다.
