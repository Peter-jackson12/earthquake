# 실제 데이터 기준선 및 제한된 개선 실험 — 2026-10-13

**완료: 개발 데이터 전체 208,480건 × 고정 5-Fold, 8개 모델 설정, 총 40회 학습·검증.**
최종 홀드아웃 52,121건은 학습·튜닝·모델 점수 계산에 사용하지 않았습니다.
이는 제출 최종 성적이 아니라 개발 CV 결과이며, 최종 모델 확정·과제 전체 완료를 뜻하지 않습니다.

## 1. 비교 조건과 범위

전처리 v1.0.1의 `preprocess.py`를 바이트 변경 없이 사용했습니다. 입력 38개, 각 fold에서 train 166,784건 / validation 41,696건입니다.
원본·개발 데이터·split manifest는 기존 해시와 같고, 모델·실험별 같은 건물에 같은 fold를 적용했습니다.
인코더와 LR 수치 5개의 스케일러는 각 train에서만 fit했습니다. 검증 행렬은 희소 형식을 유지합니다.
값 삭제·대치, SMOTE, 추가 피처, 지역 재분할을 하지 않았습니다. 각 run은 코드·설정·데이터 해시와 환경을 기록합니다.

별도로 실제 데이터 2,000건으로 5-Fold 실행을 확인했지만, 그 표본 점수는 아래 비교표에서 제외했습니다.
합성 데이터 150건 테스트 역시 실제 성능이 아닙니다.

## 2. 전체 개발 데이터 결과

| 실험 | 모델 | CV Macro F1 평균 | Fold 표준편차 | OOF Accuracy |
|---|---|---:|---:|---:|
| baseline_v1 | Dummy | 0.241744 | 0.000004 | 0.568913 |
| baseline_v1 | Logistic Regression | 0.683677 | 0.001071 | 0.739495 |
| baseline_v1 | Decision Tree | 0.594238 | 0.004512 | 0.692882 |
| baseline_v1 | Random Forest | 0.357704 | 0.015771 | 0.611531 |
| baseline_v1 | GradientBoostingClassifier | 0.469489 | 0.016769 | 0.649635 |
| feature_budget_v1 | GradientBoostingClassifier | 0.612109 | 0.005118 | 0.695506 |
| feature_budget_v1 | Random Forest | 0.628992 | 0.003829 | 0.708375 |
| lr_balanced_v1 | Logistic Regression | 0.665166 | 0.001406 | 0.691064 |

CV 평균은 5개 fold의 Macro F1 단순평균이고 표준편차는 ddof=1입니다. 신뢰구간이나 통계적 유의성 검정이 아닙니다.
OOF Accuracy와 등급별 지표는 검증 예측 208,480건을 합친 결과입니다. CV 평균과 pooled OOF Macro F1을 혼용하지 않습니다.

## 3. 초기 설정을 해석할 때의 주의

`configs/baseline.json`은 성능 확인 전에 정한 계산량 제한 기준선입니다. 모두 라이브러리 기본값이거나 최적 설정이라는 뜻이 아닙니다.
LR은 lbfgs/C=1/max_iter=1500, DT는 depth=12/leaf=5, RF는 100 trees/depth=20/leaf=2/max_features=sqrt,
GBC는 100 stages/learning_rate=0.1/depth=3/**max_features=sqrt**입니다. 특히 GBC의 sqrt는 라이브러리 기본값이 아닙니다.

초기 RF는 학습 Macro F1 평균도 0.363738, GBC도 0.472019로 낮았습니다. 이를 RF·GBC 모델 계열의 한계로 결론 내리지 않고,
원핫 후 약 1.26만 열에서 너무 적은 분할 후보를 보는 설정이 학습을 제한하는지 확인했습니다.
`feature_budget_v1`에서는 **각 모델의 max_features 하나만** RF sqrt→0.1, GBC sqrt→None으로 변경했습니다.
깊이·반복 수·seed·전처리·평가 행은 같았습니다. RF는 0.357704→0.628992, GBC는 0.469489→0.612109로 개선됐습니다.

이는 초기 설정의 후보 변수 수 제한이 이 데이터에서 성능에 영향을 주었다는 근거입니다. RF 깊이 제한 등 다른 요인은 아직 탐색하지 않았습니다.
현재 비교한 설정에서는 LR이 가장 높지만, 충분히 튜닝한 다른 모델보다 본질적으로 우수하거나 최종 선택됐다는 뜻은 아닙니다.

## 4. 클래스 가중치는 손익을 함께 봐야 함

`lr_balanced_v1`은 기존 LR의 `class_weight`만 None→balanced로 바꾸었습니다. 가중치는 각 학습 fold의 클래스 빈도로 계산됩니다.

| 지표 | LR 무가중치 | LR balanced |
|---|---:|---:|
| CV Macro F1 평균 | 0.683677 | 0.665166 |
| OOF 1등급 Precision | 0.660920 | 0.439465 |
| OOF 1등급 Recall | 0.505448 | 0.837256 |
| OOF 1등급 F1 | 0.572822 | 0.576390 |
| OOF 2등급 F1 | 0.788279 | 0.710546 |
| OOF 3등급 Recall | 0.638433 | 0.742024 |
| OOF 3등급 F1 | 0.689940 | 0.708563 |
| 실제 3등급→예측 1등급 비율 | 0.006693 | 0.035042 |

1등급 Recall은 약 50.54%→83.73%로 높아졌지만 Precision은 66.09%→43.95%로 낮아졌습니다.
2등급 F1 손실 때문에 Macro F1은 0.683677→0.665166으로 하락했습니다.
3등급 Recall도 높아졌지만 3등급을 1등급으로 크게 과소예측한 비율은 0.6693%→3.5042%로 증가했습니다.
따라서 이 balanced 설정을 Macro F1 기준의 개선안으로 채택하지 않습니다. 다른 가중치·다른 모델·샘플링 전체를 부정하는 결과도 아닙니다.

혼동행렬의 행은 실제 등급, 열은 예측 등급, 순서는 모두 1/2/3입니다.

```text
LR 무가중치                  LR balanced
[10159,  9527,   413]        [16828,  2798,   473]
[ 4745, 99465, 14397]        [19019, 75471, 24117]
[  467, 24761, 44546]        [ 2445, 15555, 51774]
```

## 5. 검증 증거와 한계

- 데이터 설치 환경: 기존 26개와 새 모델링 8개, **34/34 테스트 통과**.
- 코드 반영 커밋 `d3c49dfbe0e4fbc0b4b0d21e61493fa9d9106821`의 GitHub Actions run `38337784958`: **성공**, 15개 통과 / private 데이터 의존 19개 명시적 skip. CI가 실제 데이터 모델 성능을 재현한 것은 아닙니다.
- 실제 8개 설정의 40개 fold에서 수렴 경고를 포함한 학습 경고 0개. LR은 반복 횟수도 저장했습니다.
- 각 설정에서 개발 건물당 OOF 1개, 동일 ID·라벨·fold, 확률 순서와 합을 확인했습니다.
- 각 설정의 첫 fold 모델을 저장/재로딩하고 검증 표본 최대 32개의 예측·확률 일치를 확인했습니다. 전체 모델 저장물 전수 비교는 아닙니다.
- `audit_results.py`가 저장된 OOF로 혼동행렬과 F1을 직접 다시 계산하여 fold 평균·표준편차·등급별 지표를 대조했습니다. 모델을 독립 재학습한 검증은 아니며 학습 점수·시간은 실행 기록입니다.

`model_summary.csv`는 설정별 집계, `fold_summary.csv`는 40개 fold 집계입니다.
상세 실행 metadata·OOF 해시·파라미터는 `audit_results.py`가 생성하는 `result_provenance.json` 및 전달한 증거 묶음에 있습니다.
원시 CSV, split 행, 홀드아웃 정답, 건물별 예측, 학습 모델은 공개 Git에 올리지 않습니다.

## 6. 다음 단계와 미실행 범위

수업 요구사항 정리는 [프로젝트 가이드](../docs/PROJECT_GUIDE_KO.md), 실행 명령은 [실행 안내](../docs/RUNBOOK.md),
단계별 상태는 [계획](../docs/PLAN.md)을 봅니다. 현재 80/20·5-Fold는 유지한 작업 기준이며 팀의 공식 합의가 완료됐다는 뜻은 아닙니다.

개발 데이터 EDA는 이후 개인 1차 해석과 독립 재집계를 완료했습니다. [EDA 보고서](EDA_FINDINGS_20261013.md)와 [후속 작업 보고서](FOLLOWUP_REPORT_20261013.md)를 봅니다.
이 보고서의 8개 설정·40회 기준선 숫자는 과거 실행 기록으로 보존하며, 후속 결과를 소급해서 덮어쓰지 않습니다.

아래 목록은 기준선 작성 시점의 남은 범위입니다. 최신 완료 상태는 후속 보고서와 PLAN을 우선합니다.
남은 핵심: 팀 문제 정의·분할 합의, 다른 필수 모델의 가중치 비교, Random Under Sampling·SMOTENC의 fold 내부 비교,
체계적인 제한된 CV 튜닝, 최종 후보 확정, 최종 holdout 평가, Streamlit 대시보드, README·팀 Notion·개인 수행 기록.
후보 변수 수 비교는 1회 변경 실험이지 GridSearchCV/RandomizedSearchCV를 마친 것으로 세지 않습니다.
지역 그룹 CV와 SHAP은 필수 항목 이후 선택 심화입니다. 저장소 계획 문서가 예약 실행이나 응답 후 지속 실행을 설정하지는 않습니다.

## 재실행

데이터는 v1.0.1 패키지의 data/를 별도로 설치하고 실행하세요. 아래 결과 디렉터리는 비어 있어야 합니다.

```bash
python baseline.py --output outputs/baseline_full_v1 --jobs 2
python baseline.py --config configs/feature_budget.json --models gbc rf --output outputs/feature_budget_v1 --jobs 2
python baseline.py --config configs/lr_balanced.json --models lr --output outputs/lr_balanced_v1 --jobs 2
python audit_results.py --runs outputs/baseline_full_v1 outputs/feature_budget_v1 outputs/lr_balanced_v1 --output reports
```

### API 참고

- [scikit-learn 1.8 GBC: max_features](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [scikit-learn 1.8 LR: class_weight](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)

수치 근거는 참고 논문이 아니라 첨부 데이터의 실제 실행 결과입니다.
