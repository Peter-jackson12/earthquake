# T10 — 지역 계층적 Target Encoding(단일 요인) 사전 등록 계획

- 실험 ID: `T10_geo_hier_te_v1`
- 브랜치: `peter-exp/geo-hier-te-v1` (기준 `b54b129` = T8 감사 완료 상태)
- 코드: `experiments/geo_hier_te/` (`config.json`, `run.py`), 테스트: `tests/test_geo_hier_te.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T10_20261014_geo_te/` (Git 밖, 사전에 존재하지 않아야 함)
  - `control_check/` 대조 확인, `run/` 연구 실행(설계 봉인·원장·Fold 산출물·`backup/`), `logs/` 표준 출력 로그
- 이 문서·설정·실행기는 학습 전에 커밋되며, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다. 이 문서도 봉인 대상 소스다.

## 1. 동기

- T6R: 지역 열을 빼면 랜덤 CV 성능을 잃고 미관측 지역 성능을 얻는 교환관계가 관찰됐다.
- T7: geo one-hot 블록 L2 축소로는 중간 지점을 찾지 못했다.
- T8: LightGBM native categorical은 geo2 미관측 지역에서 지역 정보에 과적합했다.
- 팀원 아이디어: 지역 target encoding 9개 특성(1등급 비율, 3등급 비율, 건수 × 3 수준; 내부 5-Fold OOF, smoothing 20). 다만 RandomOverSampler와 함께 쓰였고, 전역 prior·원본 정수 ID 사용 가능성이 있었다.
- T10은 이를 **학습 Fold 안에서만 만든 계층적 TE 한 요인**으로 교정해 검증한다. 재표집·튜닝·원본 ID 투입은 하지 않는다.

## 2. 단일 요인과 TE 정의

- 단일 요인: 3개 지역 열(`geo_level_1/2/3_id`)의 표현. 각 모델 계열의 one-hot 기준선에서 **지역 one-hot 블록만** 9개 TE 특성으로 바꾼다. 나머지는 기준선과 동일하다.
  - C 연령 정책(age 995 → 학습 Fold의 non-995 중앙값 + `age_is_995`), 수치 5열 `StandardScaler`, 이진 22열 + 플래그, 텍스트 범주 8열 one-hot(`OneHotEncoder(handle_unknown='ignore')`), 모델 파라미터.
  - 원본 지역 ID는 어떤 수치 형태로도 행렬에 넣지 않는다(특성 이름·위치 검사).
- 수준 L ∈ {geo1, geo2, geo3}마다 3개 특성:
  - `te1_L` = 평활된 P(grade=1), `te3_L` = 평활된 P(grade=3), `cnt_L` = log1p(학습 데이터에서 그 지역의 건수)
- 계층적 평활(m = 20): `te = (n·rate + m·parent) / (n + m)`
  - geo1의 parent = 학습 데이터 전역 등급 비율, geo2의 parent = 소속 geo1의 평활값, geo3의 parent = 소속 geo2의 평활값.
  - 키는 합성 키를 쓴다: `geo1`, `geo1|geo2`, `geo1|geo2|geo3`(개발 데이터에서 geo2/geo3 ID는 전역 고유지만 방어적으로).
  - 학습 데이터에 없는 지역(n=0) → parent 값과 같고 `cnt=0`.
- 특성 순서: `te1_geo1, te3_geo1, cnt_geo1, te1_geo2, te3_geo2, cnt_geo2, te1_geo3, te3_geo3, cnt_geo3`. 행렬 위치 28–36(수치·이진·플래그 28열 다음), 그 뒤에 텍스트 one-hot.
- 계산은 float64, 행렬은 float32 CSR(기준선과 같은 컨테이너).

## 3. 누수 통제

- 바깥 학습 Fold 안에서 학습 행은 **내부 5-Fold OOF**로 인코딩한다: `StratifiedKFold(5, shuffle=True, random_state=42)`를 바깥 학습 Fold 라벨에 적용. 각 행의 통계·전역 prior·건수는 그 행이 속하지 않은 내부 학습 부분에서만 계산한다.
- 검증 행은 **바깥 학습 Fold 전체**의 통계로 인코딩한다.
- 검증 행 라벨과 개발 데이터 전체로는 아무것도 계산하지 않는다. 실행기의 `fit_arm`은 검증 라벨을 인자로 받지 않는다.
- 구현: `HierGeoTargetEncoder.fit_transform(X, y)` = 내부 OOF 인코딩, `transform(X)` = 전체 학습 Fold 통계(sklearn `TargetEncoder`와 같은 교차 적합 방식).
- 알려진 구조적 특성(설계 일부로 기록): geo2/geo1 Group 분할에서는 검증 행의 geo2·geo3(geo1 분할은 모든 수준)가 학습 Fold에 없으므로 검증 인코딩이 상위 수준 값으로 떨어지고 `cnt=0`이 된다. 반면 학습 행은 내부 OOF에서 대부분 자기 지역이 관측된다. 이 학습/검증 분포 차이는 설계대로 두고, Fold별 미관측 행 수·TE 평균을 기록한다.

## 4. 비교 군

| 군 | 지역 표현 | 그 외(기준선과 동일) |
|---|---|---|
| LR_C(기준선, 재학습 없음) | one-hot | `experiments/age995/run.py` `make_pipeline('C')`, lbfgs C=1, max_iter 1500, tol 1e-4, seed 42, 단일 스레드 |
| **LR_TE** | 9 TE → `StandardScaler`(학습 Fold fit) | LR_C와 동일. TE용 스케일러는 열 단위이므로 수치 스케일러에 열을 추가한 것과 수학적으로 같고, 수치 열 경로는 바이트 단위로 그대로 둔다 |
| LGBM_O(기준선, 재학습 없음) | one-hot | T4 `LGBM_C_v1`/T8 arm O: C 전처리기 + T4 LightGBM 파라미터(`n_jobs=2`) |
| **LGBM_TE** | 9 TE(스케일링 없음) | LGBM_O와 동일(수치 경로도 O와 같이 C 그대로) |

- 매 Fold에서 TE 행렬의 첫 28열과 텍스트 one-hot 블록이 같은 Fold에 fit한 C 전처리기 행렬의 해당 열과 같은지 희소 비교로 검사한다(전체 `.toarray()` 없음).
- 고정: early stopping·class weight·재표집·튜닝·임계값/결정 규칙 변경 없음(확률 argmax), 행 삭제 없음.

## 5. 분할·학습 횟수·실행 정책

- 분할: `stratified`(`split_manifest.csv` fold), `geo2`, `geo1`(T8 `run/{geo2,geo1}/manifest.csv`, SHA `36a40ddf…`, `1b4616ff…`를 run 폴더로 복사·검증) × 5 Fold.
- 신규 연구 fit: 2군 × 3분할 × 5 Fold = **30회(최대 30)**. 기준선은 재학습하지 않는다.
- Fold 순차 실행(한 단계 병렬: LR 1 스레드, LightGBM 2 스레드). 자동 재시도 0회. LR 비수렴 시 산출물을 저장·기록한 뒤 중단한다(파라미터 변경·재시도 없음).
- 기준선 OOF(읽기 전용, SHA 고정):
  - LR_C: stratified `de43f997…`, geo2 `9e58f72b…`, geo1 `45360835…`
  - LGBM_O: stratified T4 `0bc63eb5…`, geo2 T8 `ba282656…`, geo1 T8 `f4f3f9c9…`(T8 `metrics.json`·독립 감사 PASS에 고정)
- `prepare`는 LR_C 평균이 등록 임계값(stratified 0.688687, geo2 0.565146, ±5e-7)과, LR_C geo2 pooled 3등급 Recall이 0.5211(±5e-5)과 같은지 확인한다.

## 6. 대조 확인(연구 학습 아님, prepare 전 PASS 필수)

개발 데이터에서 층화 표본 5,000행을 뽑아 80/20 층화 분할한 뒤 다음을 확인한다.
- (a) 작은 합성 계층에서 손으로 계산한 기대값과 일치. 표본에서 벡터화 구현이 사전(dict) 기반 brute-force 참조 구현과 `atol=1e-12`로 일치(검증 인코딩과 내부 OOF 인코딩 모두).
- (b) 내부 OOF 행은 자기 라벨을 쓰지 않음: 내부 Fold 배정을 고정한 상태에서 한 행의 라벨(및 그 행이 속한 내부 Fold 전체의 라벨)을 바꿔도 그 행의 인코딩이 변하지 않음. 다른 내부 Fold의 인코딩은 바뀌어야 함(비공허성). 라벨을 바꾸면 StratifiedKFold 배정 자체가 달라질 수 있어 배정을 고정해 검사한다.
- (c) 검증 라벨을 뒤섞어도 두 군의 검증·학습 행렬이 변하지 않음.
- 원본 지역 열·지역 one-hot 열이 없음. 표본 학습 부분(4,000행)에서 두 군 smoke fit 각 1회(연구 학습 아님)와 C 대비 비지역 열 동일성.
- 결과: `control_check/control_check.json`.

## 7. 체크포인트

- 각 Fold 직후 저장: LR은 파이프라인 joblib, LightGBM은 모델 텍스트 + 전처리기 joblib, 검증 예측 확률 CSV(`%.17g`, round-trip), Fold JSON.
- SHA-256 원장(`ledger.json`), `attempts.jsonl`, `run.log`. 저장 직후 재로딩 검증(LR 정확히 일치, LightGBM `atol=1e-12`), `backup/`에 복사·SHA 대조.
- 재실행 시 완료 Fold는 해시만 검증하고 재학습하지 않는다. `RUNNING.lock`으로 동시 실행을 막는다.

## 8. 지표

T8과 같다: Fold Macro F1(labels [1,2,3]) 평균·SD(ddof=1), pooled 지표, 클래스별 P/R/F1, 혼동행렬, 3등급 Recall, 3→1·3→2 건수와 비율(실제 3등급 지원 수 기준), Fold 승수, `experiments/geo_ablation/audit.py` `turnover()`로 같은 행 전이(각 기준선 대비, LGBM_TE는 LR_C 대비도), age995와 geo1 지역 26/20/13 프로필, Fold별 검증 행 중 학습 Fold에 없는 geo1/geo2/geo3 수, TE 요약(prior, 수준별 관측 지역 수, 학습 OOF/검증 TE 평균), LR 수렴·반복 수·TE 계수, LightGBM TE gain 비중.

## 9. 사전 등록 판정(학습 전 고정)

계열 F ∈ {LR, LGBM}, 자기 one-hot 기준선 B(LR→LR_C, LGBM→LGBM_O).

1. **일반화 후보**: 다음을 모두 만족.
   - geo2 평균 F1(F_TE) ≥ F1(B) + 0.010, geo2 Fold 승수 ≥ 4/5(같은 Fold에서 엄격히 큼)
   - stratified 평균 F1(F_TE) ≥ F1(B) − 0.010
   - geo2 pooled 3등급 Recall(F_TE) ≥ B − 0.01
2. **기준선 도전 후보**(LR_C 대비): stratified 평균 > 0.688687이고 LR_C 대비 4/5 이상 승, geo2 평균 ≥ 0.565146이고 LR_C 대비 4/5 이상 승, geo2 pooled 3등급 Recall ≥ 0.5211. 충족해도 별도 사전 등록 확인이 필요한 후보일 뿐이며, 홀드아웃 평가·최종 모델 변경은 하지 않는다.
- 모든 결과를 보고한다. 두 TE 군을 같은 Fold에서 비교하므로 선택 시 낙관적이다. geo1 Group은 서술만 한다(모든 수준 미관측 → TE가 전역 prior로 축소). 유의성·인과 주장은 하지 않는다.

## 10. 순서

테스트 → 계획·코드 커밋 → 대조 확인 → `prepare`(봉인) → `fit-all` → `finalize`(OOF·지표·`analysis.json` 판정). 로그는 비공개 폴더에 둔다. 보고서·집계 JSON 커밋과 push는 메인 세션의 독립 감사 후에만 한다.

## 11. 실행 명령(worktree 루트, 동결 Python)

```
PY=../../local_reproduction/20261014_01/venv/Scripts/python.exe
OUT=../../local_reproduction/T10_20261014_geo_te
T8=../../local_reproduction/T8_20261014_native_cat/run
$PY -m experiments.geo_hier_te.run control-check --data ../../earthquake/data --destination $OUT/control_check
$PY -m experiments.geo_hier_te.run prepare --data ../../earthquake/data --t8-run $T8 \
    --t4-oof ../../local_reproduction/20261014_01/private_t4_original/private/model_families_v1/LGBM_C_v1/oof_predictions.csv.gz \
    --lr-oof-stratified ../../local_reproduction/20261014_01/private_original/age995_abc_v1/C/lr/oof_predictions.csv.gz \
    --lr-oof-geo2 ../../local_reproduction/T6S_20261014_c_group/t5_private/outputs/geo_group_cv_v1/geo2/C/oof.csv.gz \
    --lr-oof-geo1 ../../local_reproduction/T6S_20261014_c_group/t5_private/outputs/geo_group_cv_v1/geo1/C/oof.csv.gz \
    --output $OUT/run --code-commit <full SHA> --control $OUT/control_check/control_check.json
$PY -m experiments.geo_hier_te.run fit-all --data ../../earthquake/data --output $OUT/run
$PY -m experiments.geo_hier_te.run finalize --data ../../earthquake/data --output $OUT/run
```
