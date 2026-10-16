# T38 — DT·RF·sklearn GBC의 범위 확장 중첩 CV 튜닝 사전등록 설계 (T37 후속)

**상태: 사전등록 확정 — 승인(2026-10-17)·구현 중·미실행.** T37에서 DT·GBC·RF의 선택이 모두 탐색 범위 경계에 걸렸다. 사용자가 2026-10-17에 "범위를 넓힌 후속 튜닝(예산 포함) 제안"을 요청했고, 제안(GBC 깊이 5·7 × 600단계, J1 비교 대상 = T37 튜닝 절차, RF 잎 2 고정)에 "전부 제안대로 할게"라고 승인했다. 아래 후보·예산·판정 임계는 **학습 전에 고정**하며 결과를 본 뒤 바꾸지 않는다. 이 문서에는 실행 결과가 없다. `earthquake/data/`는 열거하지 않았고 `earthquake/data/final_holdout/`은 접근하지 않았다.

> **가이드 대조 줄 (AGENTS 필수):** 과제 가이드 체크리스트의 **필수 항목 "네 필수 모델 CV·튜닝"**(탐색 범위, 선택 파라미터, 검증 결과 기록)의 **후속 확장**이다(T37 범위 경계 보완). 불균형 처리 비교는 범위 밖이다. **주 평가(개발 Stratified 5-Fold seed 42, `split_manifest.csv`) 결과를 낸다.** 판정 분할은 바깥 층화 5-Fold 하나뿐이며, 선택은 각 바깥 학습 Fold 안의 층화 3-Fold(안쪽)만 쓴다. geo2/geo1 Group 분할은 실행하지 않는다.

- 실험 ID: `T38_wide_tuning_v1`. 브랜치 `peter`만 사용한다.
- 비공개 출력(실행 전 없어야 함): `local_reproduction/T38_20261017_wide_tuning/`(`preflight/`, `run/`, `logs/`). 독립 감사: `local_reproduction/session_20261016_claude/t38_audit/`.
- 코드: `experiments/wide_tuning/`(`config.json`, `run.py`), `tests/test_wide_tuning.py`(합성만). 공통 함수는 T37의 봉인된 `experiments/required_tuning/run.py`를 **import만** 한다. 공개 보고 후보: `reports/WIDE_TUNING_<날짜>.md`와 집계 JSON(독립 감사 PASS 뒤에만).

## 1. 설계

전처리·분할·선택 규칙은 T37과 같다. 전처리는 v1.0.1 `make_model_pipeline`(표준화 없음, 원핫)을 해당 학습 행에서만 fit한다. 안쪽 분할은 바깥 학습 Fold 안의 `StratifiedKFold(3, shuffle=True, random_state=42)`다. 선택 기준은 pooled 안쪽 Macro F1이고, 동률이면 등록 순서를 따른다. **가중치·재표집은 끈다.** LR은 제외한다(T37에서 `C`=1이 범위 안쪽에서 선택됨).

### 1.1 후보 (첫 후보 = T37에서 가장 많이 선택된 값, 동률 우선)

| 모델 | 고정 설정 | 후보 | 수 |
|---|---|---|---:|
| DT | `random_state` 42 | (`max_depth`, `min_samples_leaf`): DT0 (16,50), 이어서 `max_depth` {16, 20, 24, 32, None} × `min_samples_leaf` {5, 20, 50, 100}의 나머지 19개(깊이 오름차순, 잎 오름차순, None은 마지막) | 20 |
| RF | `n_estimators` 100, `min_samples_leaf` 2, `random_state` 42, 학습 `n_jobs` 4 / 예측 `n_jobs` 1 | (`max_features`, `max_depth`): RF0 (0.3,20), 이어서 `max_features` {0.3, 0.45, 0.6, 0.8} × `max_depth` {20, 30}의 나머지 7개 | 8 |
| GBC | `learning_rate` 0.1, `max_features` None, `subsample` 1.0, `random_state` 42 | (`n_estimators`, `max_depth`): GBC0 (300,5), GBC1 (600,5), GBC2 (300,7), GBC3 (600,7) | 4 |

### 1.2 GBC 단계 예측

- 안쪽에서는 깊이(5, 7)마다 600단계로 한 번 학습한다.
- 300단계 후보의 예측은 같은 모델의 앞 300단계 확률(`staged_predict_proba`)로 계산한다. 그래서 안쪽 GBC fit은 바깥 5 × 안쪽 3 × 깊이 2 = 30회다.
- `subsample`=1.0이고 조기 종료가 없으므로, 앞 n단계는 `n_estimators`=n으로 따로 학습한 모델과 같아야 한다. 사전 점검에서 이 동일성을 비트 단위로 확인하고, 다르면 봉인하지 않는다.
- 바깥 재학습은 선택된 후보의 `n_estimators`로 직접 학습한다.

### 1.3 바깥 단계

선택된 후보로 바깥 학습 Fold 전체를 학습해 바깥 검증을 예측한다. **기준 예측을 재사용하지 않는다**(모델당 5회, 총 15회).

### 1.4 비교 대상 OOF

| 용도 | 모델 | 경로(`local_reproduction/` 기준) | SHA-256 | Fold 평균 |
|---|---|---|---|---:|
| J1 (T37 튜닝 절차) | DT | `T37_20261016_required_tuning/run/DT/outer/oof.csv.gz` | `c4a1225ad72839394d23bfa7c0b02f2985dbda84a52cb96b836c357385290d8d` | 0.623402 |
| J1 | GBC | `T37_20261016_required_tuning/run/GBC/outer/oof.csv.gz` | `315bf2c816b8741dc2bfd6f9b6c808baa3910fadbab66b2aef23e7a540792ce9` | 0.662003 |
| J1 | RF | `T37_20261016_required_tuning/run/RF/outer/oof.csv.gz` | `d06146e71216a0c40dccdd2b38cef4970d17234fe7481d225a189c83769a8664` | 0.646608 |
| J1b (A2 기준 설정) | DT | `20261014_01/runs/baseline_initial/dt/oof_predictions.csv.gz` | `0f9b57c0853d8418929428cf294bf5e8fbb71dc63df4fbe0dfeefc9faaf0712f` | 0.594238 |
| J1b | GBC | `20261014_01/runs/feature_budget/gbc/oof_predictions.csv.gz` | `ff9d764e2cb8e5faebaffa1fc118a529d989ef5e99aea653f18d444ca8d99477` | 0.612109 |
| J1b | RF | `20261014_01/runs/feature_budget/rf/oof_predictions.csv.gz` | `b99e35c4210cc80ecb8dc0ca06a6574bc57cf4c50ebd247027b8adf7019ceb60` | 0.628992 |

**대조 학습(preflight, 연구 fit 아님, 3회)**: T37 바깥 예측을 이 실행기로 재현할 수 있는지 확인한다.

| 모델 | 다시 학습할 것 | 비교 대상 |
|---|---|---|
| DT | 바깥 Fold 0의 DT0 (16,50) | T37 DT 바깥 Fold 0 예측 |
| GBC | 바깥 Fold 0의 GBC0 (300,5) | T37 GBC 바깥 Fold 0 예측 |
| RF | 바깥 Fold 1의 RF0 (0.3, 깊이 20, 잎 2) | T37 RF 바깥 Fold 1 예측 |

세 경우 모두 T37 그 Fold의 선택 후보와 같은 설정이다. 확률 최대 절대차는 1e-9 이하여야 한다. 예측이 다른 행은 기준 확률의 1·2위 차이가 1e-9 이하인 동률 행뿐이어야 한다.

## 2. 예산·체크포인트·시간

- **안쪽 fit 450회**: DT 300, RF 120, GBC 30.
- **바깥 재학습 15회**: 모델당 5회.
- 계획 상한 **465 fit**, 시도 상한 **490 fit**. unit당 시도는 최대 2회, 수동 복구는 최대 10 unit, 자동 재시도는 0이다.
- 대조 학습 3회, 단계 동일성 점검, 모델 프로브는 예산 밖이며 판정에 쓰지 않는다.
- 실행 순서는 DT → RF → GBC이고, 각 모델 안에서 바깥 Fold 0부터 진행한다.
- 병렬은 RF 학습의 `n_jobs=4` 한 단계뿐이다. 희소 CSR을 그대로 쓴다.
- **예상 시간(미측정, T37 실측 기반)**: DT 약 15분, RF 약 1.5시간, GBC 약 3.5~4시간, 합계 **약 5~6시간**. 첫 모델을 실측한 뒤 갱신하며, 설계와 판정은 바꾸지 않는다.
- **체크포인트**는 T37과 같다.
  - unit 완료 즉시 다음을 저장한다: 파이프라인(joblib), 예측 CSV(GBC 안쪽은 후보별 2개), unit JSON.
  - SHA를 원장·`attempts.jsonl`에 기록하고 `backup/`에 복사한다.
  - 재로딩 재추론 차이는 1e-12 이내여야 한다. GBC는 단계 예측까지 재로딩으로 확인한다.
  - 완료 unit은 재학습하지 않는다. `RUNNING.lock`으로 중복 실행을 막는다.

## 3. 사전 점검 (`prepare` 전 PASS 필수)

1. 입력과 분할: 바깥 Fold를 확인하고, 안쪽 3-Fold가 바깥 학습 Fold를 서로소로 나누며 바깥 검증 행을 포함하지 않는지 5개 바깥 Fold 모두 확인한다.
2. 비교 OOF 6개: SHA, 행 정렬, Fold, 정답, 등록된 Fold 평균(1e-6)을 확인한다.
3. 대조 학습 3회: §1.4.
4. **GBC 단계 동일성**: 바깥 Fold 0 학습 행의 처음 5,000행으로 깊이 5와 7에서 확인한다. 20단계로 학습한 모델의 앞 10단계 `staged_predict_proba`가 `n_estimators`=10으로 따로 학습한 모델의 `predict_proba`와 비트 동일해야 한다.
5. 모델 프로브(예산 밖): 바깥 Fold 0 학습 행 2,000행으로 모든 후보를 작게 학습해 joblib 재로딩 차이와 반복 차이가 1e-12 이내인지 확인한다.

## 4. 지표와 사전 고정 판정

보고 지표는 T37과 같다: Macro F1, 평균·SD(ddof=1), Accuracy, 등급별 P/R/F1, 혼동행렬, Recall3, 3→1·3→2, 예측 비율, turnover.

| 판정 | 규칙 | 라벨 |
|---|---|---|
| **J1** 모델별, T38 절차 대 T37 절차 | Δ = T38 바깥 Fold Macro F1 − T37 튜닝 같은 Fold Macro F1. 평균 Δ ≥ +0.003이고 4 Fold 이상 Δ > 0 → `IMPROVED_DESCRIPTIVE`. 평균 Δ ≤ −0.003이고 4 Fold 이상 Δ < 0 → `WORSENED_DESCRIPTIVE`. 그 외 `NOT_MET`. | 기술 라벨, 채택 아님 |
| **J1b** 병기 | A2 기준 설정 대비 같은 규칙의 라벨과 Δ를 적는다. | 기술 라벨 |
| **J2** 안전 지표 | T38·T37·A2 기준의 Recall3, 3→1·3→2를 병기한다. 판정은 없다. | 없음 |
| **J3** 선택 기록 | 바깥 Fold별 선택, 후보별 안쪽 점수, 선택 후보의 안쪽 − 바깥을 기록한다. 선택이 다시 경계(DT 깊이 None 또는 32·잎 100, RF `max_features` 0.8 또는 깊이 30, GBC (600,7))에 걸리면 그 사실을 기록한다. 범위를 다시 넓히지는 않는다. | 기록 |

## 5. 해석 원칙과 한계

1. 바깥 5-Fold의 반복 사용은 선택 통계다. 독립 성능이 아니다.
2. Fold 5개는 독립 표본이 아니다. 유의성·인과를 주장하지 않는다.
3. 가중치를 껐다. A2의 가중치 RF 결과와 섞지 않는다.
4. 탐색은 여전히 제한된 격자다. 최적성을 주장하지 않는다.
5. 안쪽 학습(약 111k행)과 바깥 학습(166,784행)의 크기가 다르다.
6. 큰 모델은 3→1을 늘릴 수 있다. 이는 예상이며 관찰로만 기록한다.

## 6. 독립 감사 (공개 보고 전 필수)

T37 감사와 같은 범위로 한다: 원장·SHA, 분할 재계산, 저장 파이프라인 전체 재추론(GBC 단계 예측 포함), 선택·지표·J1·J1b 독립 재계산, 비교 OOF 불변, 홀드아웃 0. 감사 도구는 학습 담당과 다른 주체가 실행기를 import하지 않고 독립 구현한다.

## 7. 실행 순서

1. 단위 테스트(합성) → `python -m unittest discover -s tests -v` pass/skip 구분 → 계획·코드 커밋.
2. `preflight` PASS → `prepare`(봉인) → `fit-all`(DT → RF → GBC) → `finalize`.
3. 독립 감사 PASS 뒤에만 보고서·집계 JSON을 커밋하고, RESEARCH_STATUS·PLAN을 갱신하고, push한다.
