# T13 — LGBM_TE_G geo2 확인(새 독립 Group 분할) 사전 등록 계획

- 실험 ID: `T13_geo2_te_confirm_v1`
- 브랜치: `peter-exp/geo2-te-confirm-v1` (기준 `30c3949` = T11 독립 감사 PASS 상태)
- 코드: `experiments/geo2_te_confirm/` (`run.py`, `config.json`), 테스트: `tests/test_geo2_te_confirm.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T13_20261014_geo2_te_confirm/` (Git 밖, 사전에 존재하지 않아야 함)
  - `control_check/`, `run/`(설계 봉인·분할·원장·Fold 산출물·`backup/`), `logs/`
- 이 문서·설정·코드·테스트는 학습 전에 커밋되고, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다.

## 1. 동기

T11(독립 감사 PASS)에서 LGBM_TE_G는 단일 T5 geo2 Group 분할(seed 42)에서 사전 등록 지역 일반화 규칙을 경계에서만 충족했다
(vs LGBM one-hot O: ΔF1 +0.01013, 기준 +0.010; 4/5 Fold, 한 Fold −0.0008; vs LR-C +0.026, 5/5; 3등급 재현율 51.5% vs LR-C 52.1%).
같은 Fold가 여러 연구에서 재사용됐다. T13은 이 결과가 **새로운 독립 geo2 Group 분할**에서도 유지되는지 확인한다. 새 요인·튜닝은 없다.

## 2. 새 분할 (학습 전 고정)

- seed `[101, 202, 303]`. 각 seed: 개발 세트에서 `StratifiedGroupKFold(5, shuffle=True, random_state=seed)`, groups = (geo1, geo2) 합성 그룹(T5 geo2와 같은 절차, seed만 다름).
- manifest 열 `building_id,fold,geo_level_1_id,geo_level_2_id`, `lineterminator='\n'`. 비공개 저장 후 SHA 기록.
- 검증: 그룹·ID 서로소, 모든 학습·검증 Fold의 3개 클래스 각 ≥ 30행, 검증 크기 1%~50%(T5 게이트와 동일). 하나라도 실패하면 **seed를 바꾸지 않고 중단 후 보고**.
- seed 42 분할과의 차이: Fold 라벨의 adjusted Rand index(ARI)와 라벨 일치율, 새 seed 쌍 간 ARI를 설계에 기록.

## 3. 팔 (모두 새 분할에서 재학습, 원본 감사본과 동일 설정, 튜닝 없음)

| 팔 | 구현(수정 없이 재사용) |
|---|---|
| LR_C | `experiments.age995.run.make_pipeline('C')`, 단일 스레드, 비수렴 시 중단 |
| LGBM_O | T8 팔 O(= T4 LGBM one-hot), `n_jobs=2` |
| LGBM_TE_G | T11 팔: 그룹 인지 내부 Fold TE, `group_level='geo2'`, m=20, 내부 `StratifiedGroupKFold(5, shuffle, 42)`; 인코더는 `experiments.geo_te_group_inner.encoder` |

- 3 팔 × 3 seed × 5 Fold = **45회(최대 45)**, Fold 순차, 한 단계 병렬만(LGBM 2 스레드, LR 1 스레드), 자동 재시도 0, LR 비수렴 시 저장·기록 후 중단.
- 인코더·통계는 학습 Fold에서만 fit. 희소 one-hot 행렬 전체 `.toarray()` 금지. 홀드아웃 접근 없음.
- Fold마다 모델·예측 확률·fold json 저장, SHA 원장, 재로딩 검증, `backup/`, 완료 Fold 재학습 안 함(재개 가능).

## 4. 대조 확인(연구 학습 아님, prepare 전 PASS 필수)

1. 새 manifest 생성기에 seed 42를 넣으면 T5 geo2 manifest와 **바이트 동일**(SHA `36a40ddf…`).
2. seed 42 Fold 0에서 LGBM_O, LGBM_TE_G를 재학습해 T8 geo2/O, T11 geo2/LGBM_TE_G의 Fold 0 OOF 확률과 `atol 1e-9`로 일치(예측 라벨 100% 일치).
3. seed 42 Fold 0에서 LR_C를 재학습해 T5 C geo2 OOF Fold 0 대비 최대 확률 차이·라벨 일치율 보고. LR은 다른 환경에서 학습되었고 BLAS 민감 → **기술용, 게이팅 아님**.

게이팅은 1과 2. 대조 학습 3팔(1+1+1 Fold)은 연구 fit 45회에 포함되지 않는다.

## 5. 지표

seed별·15 Fold 풀링: Fold Macro F1(labels [1,2,3]), 평균, SD(ddof=1), 풀링 지표, 클래스별 P/R/F1, 정확도, 예측 비율, 3등급 재현율, 3→1/3→2 오분류 수, Fold 승수, 같은 행 turnover(`experiments/geo_ablation/audit.py turnover()`), age 995 및 geo1 26/20/13 프로파일.

## 6. 사전 등록 확인 규칙 (15 Fold = 3 seed × 5 Fold 전체 기준)

- **C1 (자기 기준선 대비)**: mean₁₅[F1(LGBM_TE_G) − F1(LGBM_O)] ≥ +0.010 **AND** LGBM_TE_G 승 ≥ 12/15 Fold **AND** 풀링 3등급 재현율 차이 ≥ −0.01 **AND** 모든 seed에서 분포 복구(|정확도 차| ≤ 0.02 AND LGBM_O와 예측 일치율 ≥ 0.80).
- **C2 (연구 기준선 LR-C 대비)**: mean₁₅[F1(LGBM_TE_G) − F1(LR_C)] > 0 **AND** 승 ≥ 12/15 **AND** 풀링 3등급 재현율(LGBM_TE_G) ≥ 재현율(LR_C) − 0.01.
- **확인된 geo2 후보**는 C1 **AND** C2일 때만. 그 경우에도: 랜덤 CV 성능은 재사용된 T10 stratified 결과(0.6882 vs LR-C 0.6887)이며, 홀드아웃 평가·최종 모델 변경 없음. 이후 단계(홀드아웃 등)는 사용자의 명시적 결정이 필요하다.
- seed별 결과와 모든 수치를 규칙 충족 여부와 무관하게 보고한다. 유의성·인과 주장 없음.

## 7. 금지·경계

홀드아웃·`final_holdout` 읽기 금지, 기존 브랜치·기존 비공개 결과 변경 금지, 원시 데이터·모델·OOF·manifest·행별 예측 Git 업로드 금지, push 금지, 보고서·집계 JSON은 메인 세션 독립 감사 전 커밋 금지. 모호하거나 막히면 설계를 바꾸지 않고 중단 후 보고.
