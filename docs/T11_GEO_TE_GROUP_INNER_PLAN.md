# T11 — 그룹 인지 내부 Fold 지역 Target Encoding(단일 요인) 사전 등록 계획

- 실험 ID: `T11_geo_te_group_inner_v1`
- 브랜치: `peter-exp/geo-te-group-inner-v1` (기준 `7956901` = T10 독립 감사 PASS 상태)
- 코드: `experiments/geo_te_group_inner/` (`encoder.py`, `run.py`, `config.json`), 테스트: `tests/test_geo_te_group_inner.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T11_20261014_te_group_inner/` (Git 밖, 사전에 존재하지 않아야 함)
  - `control_check/`, `run/`(설계 봉인·원장·Fold 산출물·`backup/`), `logs/`
- 이 문서·설정·코드·테스트는 학습 전에 커밋되고, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다.

## 1. 동기 (T10 감사 결과)

T10은 학습 행을 무작위 내부 `StratifiedKFold`로 인코딩했다. geo2/geo1 Group CV에서 학습 행은 거의 항상 자기 지역을 보지만(geo3 미관측은 Fold 합계 약 6,000행), 검증 행은 모두 미관측 지역이다(TE = 부모 값, cnt = 0). 모델이 이 입력 영역을 본 적이 없어 예측 분포가 이동했다(예: LGBM_TE geo1 2등급 Recall 97%→26%, Accuracy 57.7%→45.3%, LR-C와 예측 일치 28%). T11은 이 분포 차이만 줄인다.

## 2. 단일 요인

학습 행 인코딩용 내부 OOF 분할기만 바꾼다. 나머지는 T10과 같다.

| 바깥 분할 | 내부 분할기 | groups |
|---|---|---|
| geo2 | `StratifiedGroupKFold(5, shuffle=True, random_state=42)` | (geo1, geo2) 합성 키 |
| geo1 | `StratifiedGroupKFold(5, shuffle=True, random_state=42)` | geo1 |
| stratified | (재학습 없음) 내부 분할이 T10과 동일(무작위)하므로 T10 stratified OOF 재사용. 결과표에 재사용으로 표기 | — |

- 그대로: 계층적 TE(m=20, 합성 키, 9 특성, 미관측 → 부모 값·cnt 0, prior·건수는 내부 학습 부분만), 검증 행은 바깥 학습 Fold 전체 통계, LR_TE/LGBM_TE 모델·전처리·파라미터(T10 코드를 import해 재사용), 원본 지역 ID 미투입, 재표집·튜닝·early stopping·가중치·임계값 변경 없음.
- 내부 5 Fold를 만들 수 없으면(그룹 부족 등) `n_splits`를 바꾸지 않고 **중단 후 보고**한다.
- 인코더 클래스는 `-m` 진입점이 아닌 `experiments/geo_te_group_inner/encoder.py`에 둔다(T10은 `__main__.HierGeoTargetEncoder`로 pickle됨). 저장 산출물이 새 프로세스에서 이 모듈만 import해도 로드되어야 한다.

## 3. 학습

- 군: `LR_TE_G`, `LGBM_TE_G` × {geo2, geo1} × 5 Fold = **20회(최대 20)**. Fold 순차 실행, 한 단계 병렬만(LR 1 스레드, LightGBM 2 스레드), 자동 재시도 0, LR 비수렴 시 저장·기록 후 중단.
- 기준선·T10 OOF는 읽기 전용 고정 파일이며 재학습하지 않는다(SHA는 `config.json`과 T10 `design.json`에서 확인).
- 분할: T10 `run/{geo2,geo1}/manifest.csv`(SHA `36a40ddf…`, `1b4616ff…`)를 복사·검증한다.

## 4. 대조 확인(연구 학습 아님, prepare 전 PASS 필수)

실제 개발 데이터의 geo2/geo1 × 5 Fold에서 인코딩만 계산한다(분류기 학습 없음).
- (a) 내부 그룹이 내부 Fold 간 서로소이고 5개 Fold가 모두 비어 있지 않음. 바깥 검증 그룹은 학습 Fold와 서로소.
- (b) 학습 행 중 바깥 그룹 수준에서 미관측으로 인코딩된 비율을 보고하고 **100%**여야 함(geo2 분할은 geo2 수준, geo1 분할은 geo1 수준). T10의 같은 Fold 비율도 함께 기록.
- (c) 검증 행 인코딩이 T10의 `encode(fit_stats(train))`·T10 인코더 `transform`과 **바이트 단위로 같음**(학습 행 인코딩만 바뀔 수 있음).
- (d) 자기 라벨 독립(내부 Fold 라벨 전체·그룹 전체·한 행 라벨을 바꿔도 해당 행 인코딩 불변, 다른 Fold는 변함), 검증 라벨 독립(검증 라벨을 섞어도 두 군의 학습·검증 행렬 불변), 원본 지역 열 없음.
- (e) 저장한 인코더·전처리기를 새 `python -c` 프로세스에서 인코더 모듈만 import한 상태로 로드해 변환 결과가 같음.

## 5. 체크포인트

T10과 같다: Fold 직후 모델·예측 확률·Fold JSON 저장, SHA-256 원장, 재로딩 검증(LR 정확 일치, LightGBM atol 1e-12), `backup/` 복사, 완료 Fold 재학습 금지·재개 가능, `RUNNING.lock`. 추가로 매 Fold에서 (1) 학습 행 100% 미관측(바깥 수준), (2) 검증 인코딩이 T10과 바이트 동일, (3) 인코더 모듈 경로를 검사한다.

## 6. 지표

T10과 같다(Fold Macro F1 평균·SD(ddof=1), 클래스별 P/R/F1, 혼동행렬, 3등급 Recall, 3→1·3→2, Fold 승수, age995와 geo1 지역 26/20/13 프로필). 추가: 예측 분포(1/2/3 비율), Accuracy, 같은 행 전이(`experiments/geo_ablation/audit.py` `turnover()`)를 T10 군 대비·자기 one-hot 기준선 대비·LR-C 대비로, Fold별 학습 행 미관측 비율.

## 7. 사전 등록 판정(학습 전 고정)

계열 F ∈ {LR, LGBM}. 자기 one-hot 기준선 B = LR→LR-C, LGBM→LGBM one-hot O. T10 군 T = LR_TE / LGBM_TE.

1. **분포 복구**(주 판정, 서술적 관문): geo2와 geo1 **둘 다**에서 |Accuracy(F_TE_G) − Accuracy(B)| ≤ 0.02 이고 B와의 예측 일치율(pooled) ≥ 0.80.
2. **지역 일반화 후보**: 분포 복구 충족 AND geo2 평균 F1(F_TE_G) ≥ F1(B) + 0.010, B 대비 Fold 승수 ≥ 4/5 AND geo2 pooled 3등급 Recall ≥ B − 0.01 AND stratified 평균 ≥ B − 0.010. stratified는 재사용한 T10 값을 쓴다(LGBM_TE 0.6882 vs O 0.6698, LR_TE 0.6672 vs LR-C 0.6887).
3. **기준선 도전**(LR-C 대비): stratified 평균 > 0.688687 이고 LR-C 대비 4/5 이상 승, geo2 평균 ≥ 0.565146 이고 LR-C 대비 4/5 이상 승, geo2 3등급 Recall ≥ 0.5211. **미리 밝힘:** stratified는 T10 값을 재사용하므로 두 계열 모두 stratified 조건을 통과할 수 없다(LR_TE 0.6672, LGBM_TE 0.6882 ≤ 0.688687). 그래도 보고한다.

- 모든 결과를 보고한다. 같은 Fold를 T10·기준선과 공유하므로 선택 시 낙관적이다. 유의성·인과 주장, 홀드아웃 평가, 최종 모델 변경은 하지 않는다.

## 8. 순서와 명령

테스트 → 계획·코드 커밋 → 대조 확인 → `prepare`(봉인) → `fit-all` → `finalize`. 보고서·집계 JSON 커밋과 push는 메인 세션의 독립 감사 후에만 한다.

```
PY=../../local_reproduction/20261014_01/venv/Scripts/python.exe
OUT=../../local_reproduction/T11_20261014_te_group_inner
T10=../../local_reproduction/T10_20261014_geo_te/run
$PY -m experiments.geo_te_group_inner.run control-check --data ../../earthquake/data --t10-run $T10 --destination $OUT/control_check
$PY -m experiments.geo_te_group_inner.run prepare --data ../../earthquake/data --t10-run $T10 \
    --output $OUT/run --code-commit <full SHA> --control $OUT/control_check/control_check.json
$PY -m experiments.geo_te_group_inner.run fit-all --data ../../earthquake/data --output $OUT/run
$PY -m experiments.geo_te_group_inner.run finalize --data ../../earthquake/data --output $OUT/run
```
