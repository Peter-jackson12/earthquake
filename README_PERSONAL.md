# Earthquake — Peter 개인 분석 완결본

**최종 모델: v1.0.1 전처리 + 무가중치 Logistic Regression C=1.**
개발 CV Macro F1 **0.683677**, 개인 홀드아웃 Macro F1 **0.686771**, Accuracy **74.24%**.
19개 고유 설정 비교 후 후보를 동결하고 52,121건을 한 번 평가했다. 개인 결과이며 팀 최종 선정/제출 완료가 아니다.

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
