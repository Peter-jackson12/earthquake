# T12 — 지역 ID 없는 구조 파생 특성(단일 요인) 사전 등록 계획

- 실험 ID: `T12_struct_features_v1`
- 브랜치: `peter-exp/struct-features-v1` (기준 `7956901` = T10 감사 완료 상태)
- 코드: `experiments/struct_features/` (`features.py`, `run.py`, `config.json`), 테스트: `tests/test_struct_features.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T12_20261014_struct_features/` (Git 밖, 사전에 존재하지 않아야 함)
  - `control_check/` 대조 확인, `run/` 연구 실행(설계 봉인·원장·Fold 산출물·`backup/`), `logs/` 표준 출력 로그
- 이 문서·설정·실행기는 학습 전에 커밋되며, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다. 이 문서도 봉인 대상 소스다.

## 1. 목적과 선행 연구와의 구분

지역 ID를 전혀 쓰지 않는 구조 파생 특성이 랜덤 CV(stratified)와 지역 Group CV(geo2)를 **함께** 개선하는지 검증한다.
T3에서 이미 시도한 `age_band`, `geo_foundation`, `geo_age`, `foundation_age`와 지역 파생 특성, 연령(`age`) 파생 특성은 **포함하지 않는다**(age 995 정책과의 상호작용 회피).

## 2. 파생 특성 (학습 통계 없이 행 단위로 원본 입력에서 계산)

번들 S(구조 요약, 4개) — 이진 입력의 합:
- `n_super` = `has_superstructure_*` 11개 합
- `n_weak` = adobe_mud + mud_mortar_stone + stone_flag + mud_mortar_brick + timber + bamboo
- `n_strong` = cement_mortar_stone + cement_mortar_brick + rc_non_engineered + rc_engineered
- `n_use` = 세부 `has_secondary_use_*` 10개 합(agriculture, hotel, rental, institution, school, industry, health_post, gov_office, use_police, other). **집계 플래그 `has_secondary_use`는 제외**한다.

번들 D(크기·비율, 3개) — 수치 입력:
- `height_per_floor` = height_percentage / max(count_floors_pre_eq, 1)
- `area_x_floors` = area_percentage × count_floors_pre_eq
- `families_per_floor` = count_families / max(count_floors_pre_eq, 1)

관측 기술 통계(개발 데이터 208,480행, 공식에 영향 없음): `count_floors_pre_eq` **최솟값 1**, 최댓값 9, 0층 행 0개, 1층 행 32,370개. 따라서 `max(·,1)` 보호는 이번 데이터에서는 발동하지 않지만, 합성 경계 사례(0층, 1층)로 동작을 검사한다.

## 3. 비교 군(번들 × 계열 = 6개 arm)

| 계열 | 기준선(재학습 없음) | arm |
|---|---|---|
| LR | LR-C (`experiments/age995/run.py` `make_pipeline('C')`, lbfgs C=1, max_iter 1500, tol 1e-4, 단일 스레드) | `LR_S`, `LR_D`, `LR_SD` |
| LGBM | LGBM one-hot O (T4 `LGBM_C_v1` / T8 arm O: C 전처리기 + T4 파라미터, `n_jobs=2`) | `LGBM_S`, `LGBM_D`, `LGBM_SD` |

- 파생 열은 기존 행렬의 **뒤에 덧붙인다**(`ColumnTransformer`의 마지막 블록 `struct`, CSR hstack; one-hot 블록을 `.toarray()`하지 않음). 파생 열은 학습 Fold에 fit한 `StandardScaler`로 표준화한 뒤 float32로 캐스트한다. 두 계열 모두 동일(단일 요인 설계).
- 원본 38개 입력, 특수값, C 연령 정책(age 995 → 학습 Fold non-995 중앙값 + `age_is_995`), 모델 파라미터는 기준선과 동일하다.
- 매 Fold에서 (i) 기존 C 행렬의 모든 열이 바뀌지 않았고 파생 블록이 뒤에 붙었는지를 같은 Fold의 C 전처리기와 희소 비교로, (ii) 파생 열이 독립적인 pandas 계산을 학습 통계로 표준화한 값과 같은지를 검사한다.
- 고정: early stopping·class weight·재표집·튜닝·임계값/결정 규칙 변경 없음(확률 argmax), 행 삭제 없음.

## 4. 분할·학습 횟수·실행 정책

- 분할: `stratified`(`split_manifest.csv`), `geo2`, `geo1`(T10 run의 `{geo2,geo1}/manifest.csv` 복사, SHA `36a40ddf…`, `1b4616ff…`) × 5 Fold.
- 신규 연구 fit: 3 번들 × 2 계열 × 3 분할 × 5 Fold = **90회(최대 90)**. 기준선은 재학습하지 않는다.
- Fold 순차 실행(한 단계 병렬: LR 1 스레드, LightGBM 2 스레드). 자동 재시도 0회. LR 비수렴 시 산출물을 저장·기록한 뒤 중단한다(파라미터 변경·재시도 없음).
- 기준선 OOF(읽기 전용, SHA 고정; 경로·SHA는 `T10_20261014_geo_te/run/design.json` `baselines`에서 가져와 `config.json` SHA와 대조):
  - LR-C: stratified `de43f997…`, geo2 `9e58f72b…`, geo1 `45360835…`
  - LGBM O: stratified T4 `0bc63eb5…`, geo2 T8 `ba282656…`, geo1 T8 `f4f3f9c9…`
- `prepare`는 두 기준선의 평균·geo2 pooled 3등급 Recall이 등록 임계값과 같은지, 그리고 Fold 평균이 공개 앵커 `reports/lgbm_native_cat_20261014.json`과 1e-12로 일치하는지 확인한다.

## 5. 대조 확인(연구 학습 아님, prepare 전 PASS 필수)

- (a) 파생 특성이 손 계산과 일치: 합성 경계 사례(0층, 1층, 3층, 상부구조 11개 전부, 집계 사용 플래그만 켜진 행, 약/강 구조 혼합)와 실제 개발 데이터 몇 행(순수 파이썬 독립 구현과 비교).
- (b) 층화 5,000행 표본(80/20)에서 기존 C 행렬의 모든 열이 불변이고 파생 열만 뒤에 추가되며, 파생 열이 독립 계산과 일치(희소 비교, 두 계열 × 3 번들).
- (c) 파생 입력에 지역 열·`age`·범주형 열이 없음.
- (d) 표본 학습 부분에서 `LR_SD`, `LGBM_SD` smoke fit 각 1회(연구 학습 아님)와 산출물을 **새 프로세스**에서 로드(`__main__` 클래스 없음, 예측 일치).
- 결과: `control_check/control_check.json`.

## 6. 체크포인트

- 각 Fold 직후 저장: LR은 파이프라인 joblib, LightGBM은 모델 텍스트 + 전처리기 joblib, 검증 예측 확률 CSV(`%.17g`, round-trip), Fold JSON.
- SHA-256 원장(`ledger.json`), `attempts.jsonl`, `run.log`. 저장 직후 재로딩 검증(LR 정확히 일치, LightGBM `atol=1e-12`), `backup/`에 복사·SHA 대조.
- 재실행 시 완료 Fold는 해시만 검증하고 재학습하지 않는다. `RUNNING.lock`으로 동시 실행을 막는다.
- 사용자 정의 변환기(`StructFeatures`)는 별도 모듈 `experiments/struct_features/features.py`에 두어 joblib 산출물이 `__main__.*` 클래스를 참조하지 않게 한다.

## 7. 지표

Fold Macro F1(labels [1,2,3]) 평균·SD(ddof=1), pooled 지표, 클래스별 P/R/F1, 혼동행렬, accuracy, 3등급 Recall, 3→1·3→2 건수와 비율(실제 3등급 지원 수 기준), 자기 기준선 및 LR-C 대비 Fold 승수, `experiments/geo_ablation/audit.py` `turnover()`로 같은 행 전이(자기 기준선, LGBM arm은 LR-C 대비도), age995와 geo1 지역 26/20/13 프로필, LR 계수와 수렴·반복 수, LightGBM gain 비중(파생 열; 서술용).

## 8. 사전 등록 판정(학습 전 고정)

계열 F ∈ {LR, LGBM}, 번들 A ∈ {S, D, SD}, 자기 기준선 B(LR→LR-C, LGBM→LGBM-O). 다음을 **모두** 만족하면 "공동 개선 후보":
1. stratified 평균 F1(A) ≥ F1(B) + 0.003, 그리고 같은 Fold 기준 엄격히 큰 Fold 승수 ≥ 4/5
2. geo2 평균 F1(A) ≥ F1(B) + 0.003, 그리고 Fold 승수 ≥ 4/5
3. pooled 3등급 Recall(A) ≥ Recall(B) − 0.005 (stratified와 geo2 모두)
4. pooled 3→2 건수(A) ≤ 1.01 × 건수(B) (stratified와 geo2 모두)

- geo1 Group CV는 서술만 한다.
- **다중성 명시**: 3 번들 × 2 계열 = 6개 비교가 같은 Fold에서 이루어지므로, 가장 좋은 번들을 고르는 것은 낙관적이다. 어떤 후보도 별도의 사전 등록 확인 실험이 필요하다. 홀드아웃은 쓰지 않고, 최종 모델은 바꾸지 않으며, 유의성·인과 주장은 하지 않는다.
- 모든 결과를 보고한다.

## 9. 순서

테스트 → 계획·코드 커밋 → 대조 확인 → `prepare`(봉인) → `fit-all` → `finalize`(OOF·지표·`analysis.json` 판정). 로그는 비공개 폴더에 둔다. 보고서·집계 JSON 커밋과 push는 메인 세션의 독립 감사 후에만 한다.

## 10. 실행 명령(worktree 루트, 동결 Python)

```
PY=../../local_reproduction/20261014_01/venv/Scripts/python.exe
OUT=../../local_reproduction/T12_20261014_struct_features
T10=../../local_reproduction/T10_20261014_geo_te
$PY -m experiments.struct_features.run control-check --data ../../earthquake/data --destination $OUT/control_check
$PY -m experiments.struct_features.run prepare --data ../../earthquake/data --t10-design $T10/run/design.json \
    --manifests $T10/run --output $OUT/run --code-commit <full SHA> --control $OUT/control_check/control_check.json
$PY -m experiments.struct_features.run fit-all --data ../../earthquake/data --output $OUT/run
$PY -m experiments.struct_features.run finalize --data ../../earthquake/data --output $OUT/run
```
