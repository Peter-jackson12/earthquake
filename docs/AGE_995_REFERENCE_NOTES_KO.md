# age=995 처리: 동일 데이터 참고자료 4건

> 2026-10-13 원문·공개 코드 확인. 개인 브랜치의 후속 연구 참고문서다. 아래 처리를 실행하거나 성능 개선을 검증한 결과가 아니다. 논문·프로젝트 원문을 복제하지 않고 링크와 자체 요약만 남긴다.

## 적용 범위와 현재 정책

- 대상은 DrivenData **Richter's Predictor**의 260,601개 학습 건물, 38개 입력, 3개 피해 등급 과제다. 같은 지진의 더 큰 원본 데이터·5등급 연구와 구분한다.
- [공식 데이터 설명](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/)은 age를 정수형 건물 연령(년)으로 정의한다. 확인한 공식 설명에는 **995의 코딩 의미가 명시되어 있지 않다**. 결측 대체 코드(sentinel)나 실제 995년 건물이라고 확정하지 않는다.
- [개인 완료 정책](PERSONAL_COMPLETION_POLICY_20261013.md)의 age=0/995 보존, 원본 38개 입력·v1.0.1·저장된 개발 fold 보존 원칙을 유지한다.
- [개인 후보 동결 기록](PERSONAL_MODEL_SELECTION_20261013.md)에 선정된 모델과 최종 1회 평가 절차는 이 문서로 변경하지 않는다. 새 age 비교는 별도로 합의할 **후속 개발 CV 연구**다.
- 홀드아웃의 현재 실행 여부는 최신 실행 보고서에서 확인한다. 이미 결과를 본 홀드아웃에 맞춰 재튜닝한 뒤 같은 자료를 다시 독립 최종 평가라고 부르면 안 된다.

## 1. Sahil Ali: 중앙값 대체 + 995 표시

- 출처: [프로젝트](https://github.com/sahilali09/earthquake-damage-prediction), [학습 노트북](https://github.com/sahilali09/earthquake-damage-prediction/blob/main/earthquake_damage_model.ipynb), [app.py](https://github.com/sahilali09/earthquake-damage-prediction/blob/main/app.py).
- **실제 코드:** age==995를 age_is_missing 표시로 보존하고, age의 995를 NaN으로 바꾼 뒤 훈련 중앙값으로 대체한다. 배포 앱은 15.0을 사용한다.
- **확인한 순서:** 노트북은 train/validation/test를 70/10/20으로 나눈 다음 train에서 중앙값을 구해 나머지 분할에도 적용한다.
- **참고할 점:** 행과 나머지 입력을 유지하면서 특수값 여부를 별도로 전달한다.
- **검증 한계:** sentinel이라는 이름은 작성자의 해석이며 공식 코딩 근거가 아니다. age 처리만 바꾼 독립 비교는 확인하지 못했다. 이후 이미 대체·인코딩·스케일링한 자료를 RandomizedSearchCV에 넣으므로 내부 CV fold마다 전처리를 다시 fit하는 구조는 아니다. 외부 test 분리와 내부 CV 전처리 분리를 구분해야 한다.
- 우리 후속 구현에서는 age_is_995처럼 의미를 단정하지 않는 이름을 쓰고, 중앙값을 각 train fold에서 계산한다.

## 2. Zhang/Worley: 995 유지

- 출처: Dxzhang, *Nepal Earthquake Damage Grade Modeling* (2021), [작성자 설명](https://medium.com/@dxzhang0921/nepal-earthquake-damage-grade-modeling-367b3c2d95e2), [RF 노트북](https://github.com/sworley1/AdvML-TermProject/blob/main/AML_RF_Trees.ipynb).
- **실제 처리:** 글은 995인 1,363행을 유지한다고 명시한다. RF 노트북도 연령 삭제·대체 없이 인코딩과 StandardScaler를 적용한다.
- **작성자 근거:** 약 0.5%의 소수 행이며 삭제 시 평균·중앙값 변화가 작다는 설명이다.
- **검증 한계:** 역사구역 건물이라는 설명은 작성자의 검색 해석으로, 공식 근거를 확인하지 못했다. 글의 총 행 수 260,606도 공식 학습 규모와 다르다. RF 노트북은 전체 자료에 scaler를 fit한 후 분할하고 test 점수로 깊이 등을 검토한다.
- 유지 선택의 사례로 참고하되, 검증 절차나 995의 의미를 그대로 채택하지 않는다. 현재 v1.0.1 유지 기준선을 버려야 한다는 근거도 아니다.

## 3. Atheer: Isolation Forest로 훈련 이상치 제거

- 출처: *Classification of Nepal’s 2015 Earthquake Damage Levels with Gradient Boosting and SMOTE* (2020), [작성자 설명과 코드](https://medium.com/swlh/classification-of-nepals-2015-earthquake-damage-levels-with-gradient-boosting-and-smote-6449234ec649).
- **실제 코드:** IsolationForest(contamination=0.1)를 X_train에 fit_predict하고 -1로 표시되지 않은 행만 X_train/y_train에 남긴다. 공개된 출력은 182,420행에서 164,178행으로 18,242행 제거다.
- **범위:** age=995만의 처리가 아니라 여러 입력을 함께 본 **전체 이상치 제거**다.
- **참고할 점:** 표시된 제거 코드는 훈련셋에 적용된다.
- **검증 한계:** 10%라는 설정의 근거와 제거 전후 단독 비교는 확인하지 못했다. 희귀하지만 유효한 건물·지역까지 제거할 수 있다. 글의 Accuracy/F1을 대회 micro-F1 또는 우리 Macro F1과 동일한 지표로 간주하지 않는다.
- 이 사례를 995 삭제의 효과가 입증되었다는 근거로 쓰지 않는다.

## 4. Poudyal·Shakya, FAITH 2025: 넓은 연령 범위 삭제

- 출처: *Enhancing Earthquake Preparedness in Nepal through Machine Learning-Based Damage Prediction Models*, [논문 페이지](https://faith.futuretechsci.org/index.php/FAITH/article/view/109), [원문 PDF](https://faith.futuretechsci.org/index.php/FAITH/article/download/109/66/1466).
- **논문에 명시된 처리:** 3.6절(인쇄 p.483)은 IQR 기반 기준으로 age>55, 층수>5, 면적비율>20, 높이비율>10을 제거했다고 설명한다. 3.5.2절(p.481)은 거의 1,000년 및 1년 미만 값도 제거했다고 한다.
- **범위:** 995 단독 제거보다 훨씬 넓은 삭제다. 희귀 foundation h 제거 등 다른 선택도 함께 적용했다.
- **검증 한계:** age 처리만의 효과를 분리한 비교가 아니다. 3.1절의 30개 범주형·7개 수치형·15개 이진 입력 설명은 공식 38개 입력과 명확하게 정합되지 않는다. 성능 수치의 일관성 문제는 [기존 논문 비교 메모](PAPER_BENCHMARK_NOTES_KO.md#논문별-비교-한계)도 참고한다.
- 논문에 등장한다는 이유만으로 정상적인 55년 초과 건물이나 age=0을 오류라고 단정해 삭제하지 않는다. 코드 수준으로 재현한 사례가 아니라 원문에 보고된 방법이다.

## 후속 개발 CV 제안: A/B/C만 먼저 비교

다음은 실행 승인이나 새 전처리 정책이 아닌 **검토용 실험 설계**다. 현재 동결된 개인 모델의 최종 평가와 분리한다.

| 조건 | age 처리 | 확인할 질문 |
|---|---|---|
| A | 기존 v1.0.1의 원값 유지 | 공통 기준선 |
| B | 원값 유지 + age_is_995 표시 | 특수값을 명시적으로 구분하면 도움이 되는가? |
| C | 995만 train-fold 중앙값 대체 + age_is_995 표시 | 특수값 정보를 유지하면서 숫자 크기 해석을 없애면 달라지는가? |

- 모델·하이퍼파라미터·seed·저장 fold·검증 행을 고정하고 age 표현만 바꾼다. 결과가 같거나 나빠질 가능성도 포함한다.
- 중앙값·스케일러·인코더는 각 train fold에서만 fit한다. 전체 개발 데이터에서 미리 대체한 뒤 CV를 돌리지 않는다.
- C의 중앙값은 해당 train fold에서 age=995를 제외하고 계산하며 age=0은 기존 정책대로 유지한다. 이후 검증 fold에는 같은 변환만 적용한다.
- labels=[1,2,3]의 Macro F1, fold 평균·표준편차, 등급별 Precision/Recall/F1, 혼동행렬을 비교한다. OOF 집계와 fold 평균은 구분한다.
- 995/비995 그룹별 표본 수·오류도 확인하되, 작은 그룹의 변동성을 함께 밝힌다.
- 피해등급별 중앙값으로 age를 대체하거나 피해등급을 기준으로 이상치 삭제 대상을 정하지 않는다. 검증·홀드아웃 정답으로 전처리를 결정하지 않는다.
- 최종 홀드아웃을 이번 대안 선택에 사용하지 않는다. 원본·v1.0.1을 바꾸지 않고 별도 파생 실험으로 기록한다.

## 나머지 방법의 위치

| 방법 | 얻는 점 | 잃거나 확인해야 할 점 |
|---|---|---|
| 훈련행 삭제 | 의심 값의 학습 영향을 제거 | 다른 유효 입력도 잃고 희귀 그룹이 사라질 수 있음. 필요하다면 별도 승인된 ablation에서 train fold만 제거하고 검증 행은 전부 유지 |
| 중앙값만 대체 | 수치 크기의 영향을 줄임 | 특수값 그룹 정보가 사라짐. C처럼 indicator를 함께 두는 후보와 구분 |
| 상한절단 | 행을 유지하며 큰 수치의 영향을 제한 | 995와 일반 고령 건물을 합침. 55/100년 등을 공식 상한으로 단정하지 않음 |
| 구간화 + 995 독립 범주 | 995를 정상 연령의 거리·크기로 해석하지 않음 | 정상 연령의 세부 정보 감소. 구간을 데이터에서 구하면 train fold에서만 결정 |
| 원값 유지 | 정보 손실이 없고 기존 결과와 연속적 | 거리·스케일에 민감한 모델에서 큰 값의 영향 확인 필요 |

이번 조사에서 상한절단·구간화의 동일 데이터 실제 구현까지 검증하지는 못했다. 위 네 출처가 모든 방법을 실험했다거나 어느 한 처리가 우월함을 입증했다고 쓰지 않는다.

## 다음 작업 시 확인

1. 최신 AGENTS·개인 완료 정책·후보 동결 및 실행 보고서를 먼저 읽는다.
2. 이 문서를 참고자료로만 사용하고, 후속 실험 범위를 별도로 정한다.
3. A/B/C의 동일 조건 CV 결과가 생긴 뒤 실제 수행 내용과 근거를 기록한다.
4. 개인 연구 결과와 팀 공통 정책·최종 모델 결정을 구분한다.
