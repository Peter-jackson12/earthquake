# TE + ROS RandomForest: 개발 데이터 탐색 결과 (T6와 별도)

2026-10-14 KST | 연구 ID: TE_ROS_RF_EXPLORATORY_DEV_20261014
저장소: Peter-jackson12/earthquake | T5 기준 acd5ecdeefeff619c2fae9639f1627638de59383

## 결론과 상태

**RandomForest의 5-Fold 평균 Macro F1 = 0.700427108, SD(ddof=1) = 0.001069771, pooled Macro F1 = 0.700441185.**
5개 Fold 중 4개만 0.7 이상이고 Fold4는 **0.698739**다. 평균도 0.7을 겨우 초과하여 안정적인 0.7 이상 성능, 독립 최종 검증을 주장하지 않는다.

**이것은 T6 지역 피처 제거 C/R12/R1/R0 연구가 아니다.** 별도 탐색 실험이며 T4, T5, T6의 결과·브랜치를 변경하지 않는다.

## 실행 데이터/전처리/설정

- 개발 208,480행만 사용. development.csv SHA256 = 8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1.
- 기존 고정 Stratified 5Fold manifest SHA256 = 7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721.
- 외부 Fold당 학습166,784 / 검증41,696건. 최종 홀드아웃52,121건은 읽거나 평가하지 않음.
- 지역 geo1/geo2/geo3에서 등급1·3 OOF Target Encoding 6개 + 지역 빈도 count 3개를 생성. 총 모델 특성77개. 원본 지역 ID도 숫자로 사용.
- 내부 Stratified 5Fold에서 *내부 학습 조각*의 prior를 사용해 TE 누수 방지. m=20. ROS는 **외부 학습 집단에만**, TE 이후 적용.
- age995는 기존 C와 달리 **원본값 유지**. RF random_state=0, n_jobs=4. 원문 대비 pandas dtype 처리·TE prior·float32/threads 조정.
- 환경 기록: sklearn1.8.0 / pandas2.2.3 / numpy2.3.5 / imbalanced-learn0.14.1 / XGBoost3.1.3 / LightGBM4.6.0.

## Fold0 모델 5종 비교 (5-Fold 평균이 아님)

| 모델 | Micro F1 | Macro F1 | 실제3 Recall | 3→1 | 3→2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LogisticRegression | 0.690666 | 0.659946 | 72.318% | 445 | 3,418 |
| DecisionTree | 0.636728 | 0.582048 | 58.223% | 258 | 5,572 |
| RandomForest | **0.740910** | **0.700261** | 67.925% | 134 | 4,342 |
| XGBoost | 0.700955 | 0.675196 | 74.368% | 420 | 3,157 |
| LightGBM | 0.691673 | 0.665619 | 73.952% | 496 | 3,139 |

RandomForest 외 모델의 다른 Fold 성능은 수행·확인하지 않았다.

## RandomForest 고정 5-Fold

| Fold | Macro F1 | Micro F1 |
|---|---:|---:|
| 0 | 0.700261 | 0.740910 |
| 1 | 0.701455 | 0.740383 |
| 2 | 0.700451 | 0.739879 |
| 3 | 0.701229 | 0.739064 |
| 4 | 0.698739 | 0.739184 |

Pooled Micro F1/Accuracy = **0.739883922**; pooled 3등급 Recall = **0.673015737** (46,959/69,774).
실제3등급 3→1 오류744개, 3→2 오류22,071개 (각각 실제3등급 69,774개를 분모로 1.066%, 31.632%).

행=실제1,2,3 / 열=예측1,2,3 혼동행렬:

~~~text
[[12435,  7252,   412],
 [ 6898, 94857, 16852],
 [  744, 22071, 46959]]
~~~

Pooled 등급별 정밀도 = [0.619365, 0.763867, 0.731187]; 재현율 = [0.618687, 0.799759, 0.673016]; F1 = [0.619026, 0.781401, 0.700896].

## 기존 LR C와 참고 비교 (통제된 ablation 아님)

| 지표 | LR C 기존 Stratified CV | TE+ROS RF |
|---|---:|---:|
| Fold 평균 Macro F1 | 0.688687 | **0.700427** |
| Accuracy/Micro F1 | **0.741831** | 0.739884 |
| 3등급 Recall | 63.979% | **67.302%** |
| 3→1 건수 | **452** | 744 |
| 3→2 건수 | 24,681 | **22,071** |

Macro F1 개선(+0.011740)과 3등급 Recall 개선이 있지만 3→1이 292건 늘었다. C와 새 실험은 모델 계열·TE·ROS·age995 처리가 동시에 달라진다. 어느 요인의 인과적 기여인지 분리되지 않는다.

## 보존 증거와 검증 범위

원본 ZIP (코드2, 설명1, 집계 JSON2): earthquake_TE_ROS_development_eval_20261014.zip
ZIP SHA256 = 4d549778ca0d245847c79018e5fcf74afb39886b3f0dd4db5fdc007e8ef8f184.
개별 원본 멤버 SHA는 reports/te_ros_rf_exploratory_summary.json에 기록했다.

- 원본 개발 CSV와 split SHA 확인.
- 저장된 모든 Fold 혼동행렬에서 Macro/Micro F1 및 pooled CM·5Fold 평균·ddof1 SD·3등급 Recall 독립 산술 재계산 **일치**.
- 원본 실험에서 저장 모델과 행별 OOF/확률은 제출되지 않았으므로 **모델 재로딩·행별 감사·확률 재확인 수행 불가**. 집계 산술 확인만 증거 있음.
- 이 결과를 GitHub에 기록하는 과정의 신규 분류기 학습 **0회**. 기존 T6 실행/홀드아웃 미접근.

이 데이터는 학습에 없던 상위 지역 일반화를 검증하지 않았다. 지역 Group CV에서의 성능·통계적 우월성·외부 지진 배포 성능을 주장하지 않는다. 로컬 후속 연구는 TE/ROS/모델 계열의 효과를 각각 분리하고 Group CV를 별도 고정 실험으로 비교한다.

## 로컬 인계

[실험 README](../experiments/te_ros_rf_exploratory/README.md)에서 fetch·별도 worktree·실행 주의사항을 확인한다. T6 원래 ablation을 복구하거나 덮어쓰지 말 것.
