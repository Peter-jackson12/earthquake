# T9 — LightGBM 부분 native categorical(geo one-hot + 텍스트 범주 native) 사전 등록 계획

- 실험 ID: `T9_lgbm_partial_native_v1`
- 브랜치: `peter-exp/lgbm-partial-native-v1` (기준 `b54b129` = T8 감사 PASS 커밋, `--no-track` worktree `worktrees/t9_partial_native`)
- 코드: `experiments/lgbm_partial_native/` (`config.json`, `run.py`), 테스트: `tests/test_lgbm_partial_native.py`(합성 데이터만). T8 파일(`experiments/lgbm_native_cat/`)은 변경 없이 import만 한다.
- 비공개 출력: `local_reproduction/T9_20261014_partial_native/` (Git 밖, 사전에 존재하지 않아야 함)
  - `sample_check/` 실데이터 소표본 확인, `run/` 연구 실행(설계 봉인·원장·Fold 산출물·`backup/`), `logs/` 표준 출력 로그
- 이 문서·설정·실행기는 학습 전에 커밋되며, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다. 이 문서도 봉인 대상 소스다.

## 1. 동기 (T8, 독립 감사 PASS)

T8에서 범주형 11열 전체를 native categorical로 바꾼 N은 one-hot O보다 stratified에서 +0.0295(5/5 Fold) 좋았지만, 보지 못한 지역을 평가하는 geo2에서는 가장 낮았다(N 0.5457 < LR-C 0.5651 < O 0.5828).
질문: stratified 이득이 geo가 아닌 텍스트 범주 8열에서 오는가, 그리고 geo를 one-hot으로 둔 채 그 이득을 유지할 수 있는가?

## 2. 비교 군

신규 학습은 P 하나뿐이다. O·N·LR-C는 기존 OOF를 읽기 전용으로 재사용한다(재학습 없음).

| 군 | geo_level_1/2/3_id | 텍스트 범주 8열 | 출처 |
|---|---|---|---|
| O | one-hot | one-hot | stratified = T4 고정 OOF(`0bc63eb5…`), geo2/geo1 = T8 OOF(`ba282656…`/`f4f3f9c9…`) |
| N | native | native | T8 OOF(`26368054…`/`903ee979…`/`f91db6f8…`) |
| **P (신규)** | **one-hot (O와 동일)** | **native (학습 Fold OrdinalEncoder 코드)** | 이 연구 15 fit |
| LR-C | (선형 기준선) | | `de43f997…`/`9e58f72b…`/`45360835…` |

P의 정의: T8 `make_preprocessor('O')`(= C 전처리기: age 995 → 학습 Fold non-995 중앙값 + `age_is_995`, 수치 5열 StandardScaler, 이진 22열 + 플래그, float32)에서 범주형 경로만 둘로 나눈다.
- 텍스트 8열(`land_surface_condition`, `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`, `position`, `plan_configuration`, `legal_ownership_status`) → `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=NaN, encoded_missing_value=NaN, dtype=float32)`, **학습 Fold에서만** 어휘 학습, 학습 Fold에 없는 값 → NaN. 열 위치 28..35, `categorical_feature=[28..35]`로 선언.
- geo 3열 → O와 같은 `OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=float32)`(학습 Fold fit), 코드 열 뒤(36..)에 배치.
- 행렬은 **float32 CSR**을 유지한다(geo one-hot 약 12.5k열; 전체 `.toarray()` 금지). `zero_as_missing=false`이므로 CSR의 암묵적 0은 코드 0(유효 범주)이고, 미관측 NaN 코드는 CSR에 **명시적으로 저장**된다. 각 Fold에서 (a) 검증 NaN 코드 수 = 학습 어휘 밖 값 수, (b) CSR에 저장된 NaN 수 = 그 합계를 검사한다.
- 각 Fold에서 P의 첫 28열과 geo one-hot 블록이 같은 학습 Fold로 만든 O 행렬의 해당 열과 동일함을 희소 비교로 검사·기록한다(전처리기만 fit, 분류기 학습 아님).
- LightGBM 파라미터는 T8과 동일(T4 `LGBM_C_v1`: lr 0.05, 300 rounds, num_leaves 31, seed 42, `deterministic=true`, `force_col_wise=true`, `n_jobs=2`). 범주형 파라미터는 전달하지 않고 4.6.0 기본값(`max_cat_to_onehot=4`, `cat_smooth=10`, `cat_l2=10`, `max_cat_threshold=32`, `min_data_per_group=100`)을 쓰며 Fold 모델 텍스트의 실효값과 `categorical_feature=28,…,35`를 확인한다.
- 고정: early stopping 없음, class weight 없음, 재표집 없음, 임계값·결정 규칙 변경 없음(확률 argmax), 튜닝 없음, 행·열 삭제 없음.
- 기록할 차이: O 대비 열 순서가 바뀐다(head 28 → 텍스트 코드 8 → geo one-hot). 이는 표현 변경의 일부로 기록한다.

## 3. 분할과 학습 횟수

- 분할: `stratified`(`split_manifest.csv` fold, SHA `7673dbeb…`), `geo2`, `geo1`(T8 `run/{geo2,geo1}/manifest.csv`, SHA `36a40ddf…`, `1b4616ff…`를 run 폴더로 복사·검증) × 5 Fold.
- 신규 연구 fit: P × 3 분할 × 5 Fold = **15회(최대 15)**. 자동 재시도 0회. 실패 시 중단하고 보고한다.
- 병렬화: Fold 순차 실행, 각 fit은 `n_jobs=2`, `threadpool_limits(2)`(한 단계).
- `prepare`는 참조 OOF 9개의 SHA, 각 OOF의 Fold 평균 F1이 T8 공개 집계(`reports/lgbm_native_cat_20261014.json`)의 `cv_mean`과 1e-12 이내, LR-C geo2 pooled 3등급 recall = 0.5210823515922837을 확인한다.

## 4. 소표본 확인(연구 학습 아님, prepare 전 PASS 필수)

개발 데이터 중 stratified fold 0이 아닌 행에서 seed 20261014로 5,000행을 뽑아 4,000행 학습/1,000행 검증으로 나눈다(홀드아웃 없음). 같은 P 행렬을 (a) CSR 그대로, (b) 이 소표본 행렬만 dense로 바꿔 같은 파라미터로 각각 학습하고, 검증 확률 최대 차이 0(완전 일치), 예측 등급 동일, 모델 텍스트의 `categorical_feature`가 `28..35`이고 그 열 이름이 텍스트 8열임을 확인한다. 코드 열 일부에 NaN을 넣은 명시적 NaN 경로도 CSR/dense 일치를 확인한다. 결과는 `sample_check/sample_check.json`(분류기 2개, 연구 fit 수에 포함하지 않음).

## 5. 체크포인트

T8과 동일: 각 Fold 직후 모델 텍스트, 학습된 전처리기(joblib), 검증 예측 CSV(gzip, `%.17g`), Fold JSON 저장 → SHA-256 원장(`ledger.json`) 기록 → 모델·전처리기 재로딩 확률 차이 ≤ `1e-12`와 등급 동일, CSV round-trip 정확 일치 확인 → `backup/` 복사. 재실행 시 완료 Fold는 해시만 검증하고 재학습하지 않는다.

## 6. 지표

T8과 같다: Fold별 Macro F1(labels [1,2,3]), 평균, SD(ddof=1), pooled, 클래스별 P/R/F1, 혼동행렬, 3등급 recall, 3→1/3→2 건수와 비율(분모 = 실제 3등급 수), Fold 승수, 같은 행 turnover(`experiments/geo_ablation/audit.py` `turnover()`)를 P vs O, P vs N, P vs LR-C로 세 분할 모두 계산. age 995 집단과 geo1 지역 26/20/13 프로필, fit 초.

## 7. 사전 등록 규칙(학습 전 확정)

1. **분해(기술적):** stratified Fold 평균 F1로 P가 유지한 T8 이득 비율 = (P − O) / (N − O). geo2/geo1 값도 기술적으로만 함께 보고한다.
2. **"결합 후보"는 다음을 모두 만족할 때만:** stratified 평균 P > LR-C 0.688687 **그리고** LR-C 대비 4/5 Fold 이상 승(같은 Fold, 엄격히 큼); geo2 평균 P ≥ LR-C geo2 0.565146 **그리고** LR-C 대비 4/5 Fold 이상 승; geo2 pooled 3등급 recall(P) ≥ LR-C의 0.5210823515922837(≈0.5211). 충족해도 별도 확인 실험이 필요한 후보일 뿐이며, 홀드아웃 평가·최종 모델 교체는 하지 않는다.
- 판정과 무관하게 모든 결과를 보고한다. 같은 Fold에서 고른 결과는 낙관적이다. 유의성·인과 주장을 하지 않는다.

## 8. 실행 순서

테스트 → 계획·코드 커밋 → sample-check → prepare(전체 커밋 SHA로 봉인) → fit-all → finalize(OOF, 지표, 판정이 담긴 `analysis.json`). 보고서·집계 JSON 커밋과 push는 메인 세션의 독립 감사 후에만 한다.

```
python -m experiments.lgbm_partial_native.run sample-check --data ../../earthquake/data --destination <T9>/sample_check
python -m experiments.lgbm_partial_native.run prepare --data ../../earthquake/data --t8-run <T8>/run --t4-oof <T4 OOF> \
    --lr-oof-stratified <...> --lr-oof-geo2 <...> --lr-oof-geo1 <...> --sample <T9>/sample_check/sample_check.json \
    --code-commit <full SHA> --output <T9>/run
python -m experiments.lgbm_partial_native.run fit-all --data ../../earthquake/data --output <T9>/run
python -m experiments.lgbm_partial_native.run finalize --data ../../earthquake/data --output <T9>/run
```

홀드아웃(`data/final_holdout/`)은 어떤 단계에서도 읽지 않는다.
