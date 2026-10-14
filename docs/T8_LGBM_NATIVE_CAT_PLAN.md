# T8 — LightGBM 범주형 표현(one-hot vs native categorical) 사전 등록 계획

- 실험 ID: `T8_lgbm_native_cat_v1`
- 브랜치: `peter-exp/lgbm-native-cat-v1` (기준 `7a1742f` = T7 감사 완료 상태, T4 코드는 `a5d9ed5`에서 변경 없이 가져온 별도 커밋)
- 코드: `experiments/lgbm_native_cat/` (`config.json`, `run.py`), 테스트: `tests/test_lgbm_native_cat.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T8_20261014_native_cat/` (Git 밖, 사전에 존재하지 않아야 함)
  - `control_check/` 대조 확인, `run/` 연구 실행(설계 봉인·원장·Fold 산출물·`backup/`), `logs/` 표준 출력 로그
- 이 문서·설정·실행기는 학습 전에 커밋되며, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다. 이 문서도 봉인 대상 소스다.

## 1. 동기

T4의 one-hot LightGBM(`LGBM_C_v1`: lr 0.05, 300 rounds, num_leaves 31)은 stratified CV Macro F1 0.6698로 LR-C(0.6887)보다 낮았다.
외부 프로젝트(kuchbhi, 작성자 보고·미검증)는 native categorical LightGBM에 튜닝과 결정 규칙 조정을 **함께** 적용해 5-Fold Macro F1 0.708을 보고했다.
T8은 이 중 **범주형 표현 한 요소만** 분리한다. 튜닝·결정 규칙 조정은 하지 않는다.

## 2. 비교 군(단일 요인)

| 군 | 범주형 11열 표현 | 그 외 |
|---|---|---|
| O (one-hot) | T4 `LGBM_C_v1`과 완전히 동일: `make_pipeline('C')['preprocess']`의 `OneHotEncoder(handle_unknown='ignore')` CSR float32 | T4 config의 LightGBM 파라미터·스레드(`n_jobs=2`, `threadpool_limits(2)`)·seed 42·`deterministic=true`·`force_col_wise=true` |
| N (native) | 같은 C 전처리기에서 `OneHotEncoder`만 `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=NaN, dtype=float32)`로 교체. 어휘는 **학습 Fold에서만** 학습, 학습 Fold에 없는 범주 → NaN(LightGBM 결측). `categorical_feature=[28..38]`로 선언 | O와 동일 |

- 범주형 11열: `geo_level_1/2/3_id` + 텍스트 8열(`CATEGORICAL` 순서 그대로).
- 수치·이진 열: N은 T4와 **동일한 값**을 사용한다. 즉 C 정책(age 995 → 학습 Fold의 non-995 중앙값, `age_is_995` 플래그 추가), 수치 5열 `StandardScaler`(학습 Fold fit) float32, 이진 22열 + 플래그 float32. 트리에는 스케일링이 불필요하지만 단일 요인 원칙상 T4 값을 그대로 둔다. 각 N Fold에서 첫 28열이 같은 Fold의 O 행렬 첫 28열과 동일함을 검사·기록한다(전체 `.toarray()` 없음).
- 행렬 형식: O는 CSR, N은 dense float32 39열(범주형이 정수 코드 11열로 줄어 dense가 자연스러움). 이 컨테이너 차이는 표현 변경의 일부로 기록한다.
- LightGBM 범주형 파라미터는 전달하지 않고 4.6.0 기본값을 쓴다: `max_cat_to_onehot=4`, `cat_smooth=10`, `cat_l2=10`, `max_cat_threshold=32`, `min_data_per_group=100`. `prepare`에서 합성 400행 프로브(연구 학습 아님)로 실효값을 읽어 설계에 기록하고, 각 Fold 모델 텍스트의 실효값이 같아야 한다. 범주 수가 많은 열(geo3 등)의 범주 bin 보존은 LightGBM 기본 동작을 따르며, 모델 텍스트의 보존 범주 수를 Fold별로 기록한다.
- 고정: early stopping 없음, class weight 없음, 재표집 없음, 임계값·결정 규칙 변경 없음(확률 argmax), 튜닝 없음, 행 삭제 없음.

## 3. 분할과 학습 횟수

- 분할: `stratified`(`split_manifest.csv` fold), `geo2`, `geo1`(T6R `run/{geo2,geo1}/manifest.csv`, SHA `36a40ddf…`, `1b4616ff…`를 run 폴더로 복사·검증) × 5 Fold.
- 신규 연구 fit: N 15 + O 10(geo2, geo1) = **25회(최대 25)**. O stratified는 재학습하지 않고 T4 고정 OOF를 재사용한다.
- T4 O stratified OOF: `local_reproduction/20261014_01/private_t4_original/private/model_families_v1/LGBM_C_v1/oof_predictions.csv.gz`, SHA `0bc63eb5122afb479c41826918a9fbb43a3475351a934d847833656eec6930cf`(근거: `reports/model_families_v1/verification.json`(a5d9ed5)의 `LGBM_C_v1.oof_sha256`; `metrics.json` SHA `5d541790…`도 동일 근거로 고정).
- LR-C(연구 기준선) OOF: stratified `de43f997…`, geo2 `9e58f72b…`, geo1 `45360835…`(재학습 없음).
- 병렬화: Fold는 순차 실행, 각 fit은 T4 스레드 수(2)만 사용(한 단계 병렬).
- 자동 재시도 0회. 실패 시 중단하고 보고한다.

## 4. 대조 확인(연구 학습 아님, prepare 전 PASS 필수)

내 실행기의 O 경로로 stratified fold 0을 다시 학습하고, 검증 확률이 T4 고정 OOF fold 0과 `atol=1e-9` 이내, 예측 등급·정답 등급이 동일해야 한다. 이는 O 경로가 T4와 같음을 보인다. 결과는 `control_check/control_check.json`.

## 5. 체크포인트

각 Fold 직후 LightGBM 모델 텍스트, 학습된 전처리기(joblib), 검증 예측 CSV(gzip, `%.17g`), Fold JSON을 저장하고 SHA-256을 원장(`ledger.json`)에 기록한다. 저장 직후 모델·전처리기 재로딩 확률 차이 ≤ `1e-12`와 등급 동일, CSV `float_precision='round_trip'` 재읽기 정확 일치를 검사하고 `backup/`에 복사한다. 재실행 시 완료 Fold는 해시만 검증하고 재학습하지 않는다.

## 6. 지표

Fold별 Macro F1(labels [1,2,3]), 평균, SD(ddof=1), pooled, 클래스별 P/R/F1, 혼동행렬, 3등급 recall, 3→1/3→2 건수와 비율(분모 = 실제 3등급 수), Fold 승수, 같은 행 turnover(`experiments/geo_ablation/audit.py` `turnover()`)를 N vs O, N vs LR-C로 세 분할 모두 계산. age 995 집단과 geo1 지역 26/20/13 프로필, fit 초.

## 7. 사전 등록 판정 규칙(학습 전 확정)

1. **주 판정(표현 효과):** stratified 평균 F1(N) ≥ F1(O) + 0.005 **그리고** stratified 5 Fold 중 4 Fold 이상에서 N > O(같은 Fold, 엄격히 큼)이면 N은 "표현 개선"이다.
2. **보조 판정(연구 기준선 대비, 기술적 관문):** stratified 평균 F1(N) > LR-C 0.688687 **그리고** LR-C 대비 4/5 Fold 이상 승 **그리고** geo2 평균 F1(N) ≥ LR-C geo2 평균 0.565146이면 N은 "기준선 도전 후보"다. 이 경우에도 별도 사전 등록 확인 실험이 필요한 후보일 뿐이며, 홀드아웃 평가·최종 모델 교체는 하지 않는다.
- Group CV(geo2, geo1)의 N vs O는 기술적으로만 보고한다. 같은 Fold에서 두 군을 비교한 것이며 유의성·인과 주장을 하지 않는다.

## 8. 실행 순서

테스트 → 계획·코드 커밋 → control-check → prepare(전체 커밋 SHA로 봉인) → fit-all → finalize(분할·군별 OOF, 지표, 판정이 담긴 `analysis.json`). 보고서·집계 JSON 커밋과 push는 메인 세션의 독립 감사 후에만 한다.

```
python -m experiments.lgbm_native_cat.run control-check --data ../../earthquake/data --t4-oof <T4 OOF> --destination <T8>/control_check
python -m experiments.lgbm_native_cat.run prepare --data ../../earthquake/data --manifests <T6R run> --t4-oof <T4 OOF> \
    --lr-oof-stratified <...> --lr-oof-geo2 <...> --lr-oof-geo1 <...> --control <T8>/control_check/control_check.json \
    --code-commit <full SHA> --output <T8>/run
python -m experiments.lgbm_native_cat.run fit-all --data ../../earthquake/data --output <T8>/run
python -m experiments.lgbm_native_cat.run finalize --data ../../earthquake/data --output <T8>/run
```

홀드아웃(`data/final_holdout/`)은 어떤 단계에서도 읽지 않는다.
