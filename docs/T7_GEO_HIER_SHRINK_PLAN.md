# T7 — 지역 변수 계층적 축소(Hierarchical Shrinkage) 사전 등록 계획

- 실험 ID: `T7_geo_hier_shrink_v1`
- 브랜치: `peter-exp/geo-hier-shrink-v1` (기준 커밋 `0bee98e`, T6R 감사 완료 상태)
- 코드: `experiments/geo_hier_shrink/` (`config.json`, `run.py`), 테스트: `tests/test_geo_hier_shrink.py`
- 비공개 출력: `local_reproduction/T7_20261014_geo_shrink/run` (Git 밖, 사전에 존재하지 않아야 함)
- 이 문서는 학습 전에 커밋되며, 그 커밋 SHA로 `prepare`가 설계를 봉인한다. 이 문서 자체도 봉인 대상 소스에 포함된다.

## 1. 동기와 가설

T6R(독립 감사 완료) 결과: 랜덤(stratified) CV Macro F1은 C 0.689 > R12 0.653 > R1 0.593 > R0 0.458이지만,
미관측 2차 지역(geo2 Group CV)에서는 R1(geo1만 사용)이 C보다 높았고(0.584 vs 0.565, 5/5 Fold),
미관측 1차 지역(geo1 Group CV)에서는 R0가 C보다 높았다.

가설: C는 모든 one-hot 계수에 같은 약한 L2(C=1)를 적용하므로, 세분 지역(geo3/geo2) 계수가 신호를 흡수하고
상위 지역(geo1) 계수는 덜 학습된다. 미관측 세분 지역에서는 상위 계수만 남으므로 일반화가 약해진다.
세분 수준에 더 강한 축소를 걸면 신호가 상위 수준으로 이동해, 랜덤 CV 성능 대부분을 유지하면서
미관측 지역 일반화가 개선될 수 있다. 이는 검증할 가설이며 인과 주장이 아니다.

## 2. 처리(단일 요인)

- C(`experiments/age995/run.py` `make_pipeline('C')`)와 동일하되, 하나의 범주형 OneHotEncoder를
  geo1/geo2/geo3/text 네 개의 OneHotEncoder(같은 옵션: `handle_unknown='ignore'`, sparse, float32)로 나눈다.
  열 순서는 C와 동일하다(geo1 → geo2 → geo3 → text).
- `ColumnTransformer.transformer_weights`로 geo 블록을 s배 한다. one-hot 블록을 s배 하는 것은
  그 블록의 L2 벌점을 1/s²배 하는 것과 같다.
- 수치·이진 경로, LR 하이퍼파라미터(C=1, lbfgs, max_iter=1500, tol=1e-4, random_state=42, 단일 스레드),
  age995 처리(학습 Fold 중앙값), 분할은 변경하지 않는다.

| 변형 | geo1 | geo2 | geo3 | text |
|---|---|---|---|---|
| H3_050 | 1 | 1 | 0.5 | 1 |
| H3_025 | 1 | 1 | 0.25 | 1 |
| H3_010 | 1 | 1 | 0.1 | 1 |
| H23_050 | 1 | 0.5 | 0.5 | 1 |
| H23_025 | 1 | 0.25 | 0.25 | 1 |
| H23_010 | 1 | 0.1 | 0.1 | 1 |

## 3. 대조 확인(연구 학습 실행 아님)

- 단위 테스트(합성 데이터): 모든 가중치 1인 분할 OHE 파이프라인의 인코딩 행렬이 C와 같은 열 순서로 정확히 같고,
  예측 라벨이 같으며 확률 차이가 atol 1e-6 이내임을 확인한다. 가중치가 대상 블록에만 적용되는지도 확인한다.
- 실데이터 표본 확인(`control-check`): 개발 데이터에서 층화 추출한 5,000행(학습 4,000 / 검증 1,000)으로
  C와 가중치 1 분할 OHE를 각각 fit해 행렬·예측·확률(atol 1e-6)을 비교한다. **표본 확인이며 전체 CV가 아니다.**
  분류기 2회 fit은 연구 fit 예산(90)에 포함하지 않는다. PASS여야 `prepare`가 진행된다.

## 4. 검증 설계

- 분할: stratified(`split_manifest.csv` fold), geo2·geo1 Group(T5 manifest, SHA `36a40ddf…`/`1b4616ff…`, 출력 경로로 복사 후 SHA 확인).
- 6 변형 × 3 분할 × 5 Fold = **신규 fit 90회**(최대 90). 자동 재시도 0회, 비수렴 시 중단(파라미터 변경 없음).
- 병렬화: Fold 프로세스 최대 4개 동시, 각 1 스레드(병렬 단계는 하나뿐).
- C는 재학습하지 않는다. 비교 기준은 기존 C OOF(읽기 전용):
  stratified `de43f997…`, geo2 `9e58f72b…`, geo1 `45360835…`.
- Fold 체크포인트: Fold 완료 직후 모델·예측 확률·지표·로그 저장 → SHA-256 원장 기록 → 재로딩 검증 → 백업.
  완료 Fold는 재학습하지 않으며 중단 후 재개 가능. Fold CSV는 `float_precision='round_trip'`으로 읽는다.
- 홀드아웃은 읽지 않는다. 최종 모델은 바꾸지 않는다.

## 5. 지표

- 주지표: labels=[1,2,3] Macro F1의 5 Fold 비가중 평균, SD(ddof=1).
- 보조: pooled OOF Macro F1, 클래스별 Precision/Recall/F1, 혼동행렬, 3등급 Recall,
  3→1·3→2 건수와 비율(분모는 실제 3등급 수), Fold별 C 대비 승수(같은 Fold에서 엄격히 큰 경우),
  세 분할 모두에서 C OOF 대비 같은 행 전이(`experiments/geo_ablation/audit.py` `turnover()`).
- 기술적 진단(판정에 사용하지 않음): 블록별 계수 L2 노름(인코딩 단위·원래 지시변수 단위).

## 6. 사전 등록 해석 규칙

변형은 다음을 **모두** 만족할 때만 "후속 후보"로 표시한다.

1. geo2 Group CV 평균 Macro F1 ≥ C + 0.010
2. geo2 Group CV에서 C 대비 Fold 승수 ≥ 4/5
3. stratified CV 평균 Macro F1 ≥ C − 0.010

- 6개 변형은 결과와 무관하게 모두 보고한다.
- 같은 CV Fold로 6개 중에서 고르는 것은 낙관적 편향이 있으며 이를 명시한다. 후속 후보는 확정이 아니라
  별도 사전 등록 검증이 필요한 후보일 뿐이다.
- 홀드아웃 평가, 최종 모델 교체, 인과 주장은 하지 않는다. geo1 Group 결과는 보고하되 판정 규칙에는 쓰지 않는다.

## 7. 실행 순서

```bash
# worktree 루트에서, 동결 venv 사용
python -m unittest discover -s tests -v
python -m experiments.geo_hier_shrink.run control-check --data ../../earthquake/data --destination <PRIV>/control_check
python -m experiments.geo_hier_shrink.run prepare --data ../../earthquake/data --output <PRIV>/run \
  --manifests <T6R>/run --control <PRIV>/control_check/control_check.json --code-commit <FULL_SHA> \
  --c-oof-stratified <C_STRAT_OOF> --c-oof-geo2 <C_GEO2_OOF> --c-oof-geo1 <C_GEO1_OOF>
python -m experiments.geo_hier_shrink.run fit-all --data ../../earthquake/data --output <PRIV>/run --workers 4
python -m experiments.geo_hier_shrink.run status --output <PRIV>/run
python -m experiments.geo_hier_shrink.run finalize --data ../../earthquake/data --output <PRIV>/run
```

`fit-all`은 완료되면 OOF·변형별 지표·`analysis.json`을 만든다. 중단 시 같은 명령으로 재개한다.

## 8. 공개 범위

- 모델·OOF·manifest·로그·행 단위 예측은 공개 Git에 올리지 않는다.
- 보고서와 집계 JSON은 독립 감사를 통과한 뒤에만 이 브랜치에 커밋한다. 이 계획 커밋에는 결과가 없다.
