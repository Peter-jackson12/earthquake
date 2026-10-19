# 네팔 지진 건물 피해 등급 예측 — Streamlit 앱

## 폴더 구성

```
nepal_streamlit/
├── app.py              # Streamlit 앱 (5개 페이지)
├── features.py         # 실험 ③ 파생변수 93개 + 새 건물용 변환기(FeatureBuilder)
├── train_model.py      # 최종 모델 학습 (HGB + LightGBM soft voting)
├── results.py          # 노트북 실행 결과 (표·그래프용 숫자)
├── assets/             # 노트북에서 가져온 그림 (학습곡선, SHAP summary)
├── .streamlit/config.toml   # 색상 테마
└── requirements.txt
```

## 실행 방법

1. 아래 두 파일을 `app.py`와 같은 폴더에 복사합니다.
   - `nepal_building_damage.csv` (필수)
   - `nepal_building_damage_dictionary.xlsx` (선택, 컬럼 정의서 탭에 표시)
2. 패키지를 설치합니다.
   ```
   pip install -r requirements.txt
   ```
3. 최종 모델을 학습합니다 (2~5분, 처음 한 번만).
   ```
   python train_model.py
   ```
   앱의 '피해 등급 예측' 페이지에서 **최종 모델 학습하기** 버튼을 눌러도 됩니다.
4. 앱을 실행합니다.
   ```
   streamlit run app.py
   ```

## 페이지

| 페이지 | 내용 |
|---|---|
| 개요 | 피해 등급 분포, 최종 결과, 단계별 성능 변화 |
| 데이터 탐색 | 피해 등급·나이·수치형·재료·용도·범주형·상관관계 그래프, 컬럼 정의서 (대지역 필터) |
| 모델링 과정 | 베이스라인 → 파생변수 실험 ①②③ → 튜닝 → Soft voting 조합 비교 |
| 최종 모델 평가·해석 | 테스트 성능·혼동행렬, 학습곡선, 변수 중요도, SHAP |
| 피해 등급 예측 | 건물 정보를 넣거나 테스트 건물을 불러와 예측, 등급별 확률과 예측 이유(SHAP) |

## 참고

- '모델링 과정'과 '최종 모델 평가·해석'의 숫자는 노트북 실행 결과(`results.py`)입니다.
- '피해 등급 예측'은 `train_model.py`로 학습한 모델을 씁니다. 노트북 10번 최종 테스트와 같은 설정
  (class_weight 1.9 / 1.0 / 1.3, LightGBM 트리 368개, HGB 트리 250개, 학습+검증 85%)이라
  테스트 Macro F1이 0.7158 근처로 나옵니다.
- 새 건물의 지역 통계·피해 비율은 학습 데이터로 만든 표에서 가져옵니다(`FeatureBuilder`).
  `train_model.py`가 이 방식이 노트북의 `make_features` 결과와 같은지 자동으로 확인합니다.
- 학습 데이터에 없는 소지역 ID를 넣으면 지역 정보 없이 평균값으로 예측합니다.
