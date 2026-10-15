# T15 — 3등급 과소예측 결정 규칙(단일 로짓 오프셋) 사전 등록 계획

- 실험 ID: `T15_GRADE3_DECISION_RULE` (`study_id`: `T15_grade3_decision_rule_v1`)
- 브랜치: `peter-exp/grade3-decision-rule-v1` (기준 `af261c7` = T13 보고서 커밋)
- 코드: `experiments/grade3_decision_rule/` (`run.py`, `config.json`), 테스트: `tests/test_grade3_decision_rule.py`(합성 데이터만)
- 비공개 출력(Git 밖, 사전에 존재하지 않아야 함): `local_reproduction/T15_20261015_grade3_rule/` (`run/`: 설계 봉인·원장·내부 Fold 산출물·`backup/`)
- 이 문서·설정·코드·테스트는 학습 전에 커밋되고, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다.
- 상태: 계획·실행기·테스트만 작성됨. **실데이터 학습은 수행하지 않았다.**
- 확정(2026-10-15, 사용자가 메인 세션 추천안에 위임): §8 판정 기준값과 `b` 그리드(0~2.0, 0.1 간격)를 초안 그대로 확정. geo1은 진단용으로 **포함**(총 450회). class_weight는 이번에 제외하고 필요 시 별도 실험(T16 이후)으로 사전 등록.

## 1. 가설과 요인

- 가설: 3등급은 모든 방법에서 과소예측(3→2 오류 우세)되므로, 3등급 확률에 고정 배수 `exp(b)`를 곱하는 결정 규칙이 Macro F1과 3등급 재현율을 올린다. 일부 모델(특히 LGBM_TE_G)은 3→1 오류가 이미 많아 이득이 제한될 수 있다.
- **단일 요인 = 결정 규칙만.** 모델 학습 설정(파라미터·특성·분할·class weight)은 바꾸지 않는다. 각 팔은 **자기 자신의 보정 전 argmax**와 비교한다.
- 규칙: `pred = argmax_k(p_k · w_k)`, `w = (1, 1, exp(b))`.
- 파라미터 공간(사전 고정, 1개): `b ∈ {0.0, 0.1, …, 2.0}` (21개, 음수 없음). `b = 0`이 보정 없음.
- `b` 선택 기준: 바깥 학습 행의 **내부 OOF 풀링 행**에서 labels=[1,2,3] Macro F1 최대. 정확히 동률이면 **가장 작은 b**.
- class_weight(학습 단계 변경)는 모델을 다시 학습해야 해 **이번 실험에서 제외**한다(별도 요인, 필요 시 T16 이후에서 사전 등록).

## 2. 팔(3개, 규칙을 동일하게 적용)

| 팔 | 보정 전 기준(재사용) | 내부 Fold 학습 코드(수정 없이 import) |
|---|---|---|
| LR_C | 저장 OOF | `experiments.geo2_te_confirm.run.fit_one('LR_C')` = `age995.make_pipeline('C')` |
| LGBM_O | 저장 OOF | `fit_one('LGBM_O')` = T8 팔 O(= T4 one-hot) |
| LGBM_TE_G | 저장 OOF | Group 분할: T11 `fit_arm('LGBM_TE_G', geo2/geo1)`; stratified: 재사용 OOF가 T10 `LGBM_TE`이므로 T10 `fit_arm('LGBM_TE')` |

모델 파라미터·스레드는 T13/T11과 동일(`config()`가 T13 설정과 일치 여부를 `require`로 검사).

## 3. 분할과 manifest (모두 SHA-256 확인 완료, 경로는 `config.json`)

| scheme | 종류 | manifest | SHA-256 | 역할 |
|---|---|---|---|---|
| `geo2_s42` | geo2 Group | `local_reproduction/T6R_20261014_local/run/geo2/manifest.csv` | `36a40ddf1153b7a25bf36d5573b26b4dc9a0d67d240869c2e5cc1d85c1f89d64` | 주 |
| `geo2_s101` | geo2 Group | `local_reproduction/T13_20261014_geo2_te_confirm/run/manifests/seed_101/manifest.csv` | `7877b21b59433dea9c4a22ddd8fc1112a3440180fc4e2ab9a73aa0a5b56af1f4` | 주 |
| `geo2_s202` | geo2 Group | `…/seed_202/manifest.csv` | `9b3a48054743a95b83d85d95ac05354456d441fe4ac6c8aa189434d3877914c7` | 주 |
| `geo2_s303` | geo2 Group | `…/seed_303/manifest.csv` | `26f42c0b2b3ea40f736d82d178b864ce2d4a25d1eeb2d924c65a177254b8fafd` | 주 |
| `strat_s42` | stratified | `earthquake/data/split_manifest.csv` | `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721` | 보조(**낙관적**: 여러 연구가 반복 사용한 Fold) |
| `geo1_s42` | geo1 Group | `local_reproduction/T6R_20261014_local/run/geo1/manifest.csv` | `1b4616ff7d5e052ac76707e58de48e3d0b23fb7a49e350c14d6704cc1ee488f5` | 보조(진단 전용, 판정 게이트 없음) |

## 4. 누수 없는 튜닝 프로토콜

문제: 임계값/가중치를 평가 Fold 라벨로 정하면 누수다. **다른 바깥 Fold들의 OOF로 정하는 것도 누수**다 — 그 OOF 모델들은 평가 Fold의 행을 학습에 썼다.

프로토콜(바깥 Fold f마다, 팔마다):

1. 바깥 학습 행 `A_f`(= 평가 Fold f 제외)만 취한다. 평가 행 `B_f`와 그 라벨은 이 단계에 입력되지 않는다.
2. `A_f` 안에서 **내부 5-Fold**를 만든다. Group scheme은 **바깥 분할과 같은 그룹 정의**(geo2: (geo1,geo2) 합성, geo1: geo1)의 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, stratified는 `StratifiedKFold(5, shuffle=True, random_state=42)`. 내부 Fold는 `A_f`의 함수일 뿐이며 SHA를 설계에 봉인한다.
3. 내부 Fold마다 해당 팔을 `A_f`의 내부 학습 행에서 학습하고 내부 검증 행 확률을 얻는다 → `A_f` 전체에 대한 **내부 OOF 확률**.
4. 내부 OOF(라벨·확률)만으로 `b`를 선택한다 (`decide(y_inner, p_inner, p_outer, grid)` — 바깥 라벨은 인자가 아니다).
5. 이미 저장된 **바깥 모델의 평가 Fold 확률**(`A_f` 전체로 학습)에 `exp(b)`만 곱해 보정 예측을 만든다. 바깥 라벨은 `b`가 확정된 **뒤** 지표 계산에만 쓴다.
6. 실행 중 검사: 내부 OOF 행 집합 = `A_f` 정확히 일치, `B_f`와 교집합 0(아니면 중단); 내부 Fold의 그룹 서로소·클래스 지지 ≥ 30 게이트(실패 시 seed 교체 없이 중단); 내부 Fold SHA가 봉인값과 같아야 함; `b=0` 보정이 저장 `y_pred`를 그대로 재현해야 함.

이 방식이 누수 없는 이유: `b`에 쓰인 확률은 `A_f` 부분집합으로 학습한 모델이 `A_f` 다른 부분(그룹이 겹치지 않음)에 대해 낸 값이고, 라벨도 `A_f`뿐이다. `B_f`의 행·라벨·지역은 어디에도 들어가지 않는다. 내부 Group 분할은 배포 시 "처음 보는 지역"에 가까운 확률 분포를 재현한다.

한계(보고서에 명시): ① 내부 모델은 `A_f`의 약 80%(개발 데이터의 약 64%)로 학습돼 바깥 모델(`A_f` 전체)과 보정 정도가 조금 다를 수 있다. ② LR-C 바깥 OOF는 다른 환경에서 학습됐다(BLAS 민감, T13 대조에서 최대 확률 차 0.0147, 라벨 일치 99.8%); 내부 LR은 현재 동결 환경에서 학습한다. ③ stratified seed 42와 geo2 seed 42는 이전 연구에서 반복 노출된 Fold다(handoff §4) — 규칙 선택은 그 Fold 평가와 무관하게 내부에서만 이뤄지지만, 모델·설정 자체는 노출되었다.

## 5. 재사용 산출물 (읽기 전용, SHA-256 확인 완료; Fold 정렬·y_true 일치·`argmax(p) == y_pred` 100% 확인)

바깥 모델은 **재학습하지 않는다**(바깥 확률 = 아래 OOF의 평가 Fold 행). 모든 경로는 `local_reproduction/` 기준.

| scheme | 팔 | 경로 | SHA-256 |
|---|---|---|---|
| strat | LR_C | `20261014_01/private_original/age995_abc_v1/C/lr/oof_predictions.csv.gz` | `de43f9974e536b3a9c5ebac35b984ed886ba2d5426d59dda989ca12f366a2cbc` |
| strat | LGBM_O | `20261014_01/private_t4_original/private/model_families_v1/LGBM_C_v1/oof_predictions.csv.gz` | `0bc63eb5122afb479c41826918a9fbb43a3475351a934d847833656eec6930cf` |
| strat | LGBM_TE_G(=T10 LGBM_TE) | `T10_20261014_geo_te/run/stratified/LGBM_TE/oof.csv.gz` | `3c0d2cd8b3d40acc42817fe029c70335f4f5233562cfd857836d6275bd0eb859` |
| geo2 s42 | LR_C | `T6S_20261014_c_group/t5_private/outputs/geo_group_cv_v1/geo2/C/oof.csv.gz` | `9e58f72ba93d1a886ebcef345a5d8be9fac991097aee87dcb9841e31443b00b4` |
| geo2 s42 | LGBM_O | `T8_20261014_native_cat/run/geo2/O/oof.csv.gz` | `ba282656c482264c045865de32db32d56a0b7c6c872670972c480a90abaa211c` |
| geo2 s42 | LGBM_TE_G | `T11_20261014_te_group_inner/run/geo2/LGBM_TE_G/oof.csv.gz` | `34f23ad64d9dbcc3c79b3112808db65d5e8a3a114b19e086d2b1bf1756ee039b` |
| geo1 s42 | LR_C | `T6S_20261014_c_group/t5_private/outputs/geo_group_cv_v1/geo1/C/oof.csv.gz` | `453608352590cf27ddb1460926b7578d50832baa9127431422d0dd69b8f42951` |
| geo1 s42 | LGBM_O | `T8_20261014_native_cat/run/geo1/O/oof.csv.gz` | `f4f3f9c9b966394c8d1ef0d7e6b1b5fd1367390b832bdd7f49ea8b2340b3d9f1` |
| geo1 s42 | LGBM_TE_G | `T11_20261014_te_group_inner/run/geo1/LGBM_TE_G/oof.csv.gz` | `94f06d8e5ccafe33abe5c4ed0a59be11a7633769097dd497dc3fdde82bfc9993` |
| geo2 s101 | LR_C / LGBM_O / LGBM_TE_G | `T13_20261014_geo2_te_confirm/run/s101/{arm}/oof.csv.gz` | `486f7975…` / `bd2f5f17…` / `5bc7df13…` |
| geo2 s202 | 〃 | `…/s202/{arm}/oof.csv.gz` | `1c8ef79e…` / `45d0d0bb…` / `2fa8f729…` |
| geo2 s303 | 〃 | `…/s303/{arm}/oof.csv.gz` | `7dc09af6…` / `a3666891…` / `0b8ff718…` |

(전체 64자 SHA는 `config.json`에 있다. 저장된 팔 모델 파일(.txt/.joblib)은 이번 실험에서 읽지 않는다.)

## 6. 학습 횟수·예상 시간·병렬화

- **바깥 모델 재학습 0회.** 새 학습은 내부 Fold fit만: 6 scheme × 3 팔 × 5 바깥 Fold × 5 내부 Fold = **450회(최대 450)**. 주 지표용(geo2 4 scheme)만이면 300회, 보조(strat, geo1) 150회. 원장 키는 `scheme/arm/f{outer}/i{inner}`.
- 예상 시간(추정, 미측정): T13의 팔별 fit 시간(LR_C 약 14s, LGBM_O 약 9s, LGBM_TE_G 약 5s + TE 전처리)과 내부 학습 행 비율 64%를 반영하면 450회 순차 약 **1.5시간(범위 1~2.5시간)**. 첫 scheme의 첫 Fold 로그로 실제 속도를 확인하고, 크게 벗어나면 보고한다.
- 디스크 추정: 내부 모델 450개(LGBM 약 3.5MB, LR 약 0.3MB) + 백업 사본 ≈ 3GB 이내(비공개).
- **한 단계 병렬만**: Fold 순차, LGBM `n_jobs=2`(threadpool 2), LR 단일 스레드. 외부 병렬·중복 학습 없음. 자동 재시도 0, LR 비수렴 시 저장·기록 후 중단. 희소 행렬 전체 `.toarray()` 없음.
- 체크포인트: 내부 fit마다 모델·내부 확률·json 저장 → SHA-256 원장 → 재로딩 검증(LR 정확 일치, LGBM 확률 차 ≤ 1e-12, CSV 왕복 정확) → `backup/` 복사. 5개 내부 fit이 끝나면 결정 단계(`decision_f*.json`, `adjusted_f*_pred.csv.gz`)도 같은 방식으로 저장. 완료 단위는 재학습 없이 해시만 검증하고 재개한다.

## 7. 지표

바깥 Fold별·풀링: labels=[1,2,3] Macro F1(주), 클래스별 P/R/F1, 혼동행렬, 3등급 재현율, 3→1·3→2 오류 수, 예측 비율, 선택된 `b`(Fold별 분포, `b=0` 비율, 그리드 상한(2.0) 도달 비율), 내부 OOF 곡선. 바깥 Fold의 `b` 전체 스윕 곡선은 **사후 기술용**이며 선택에 쓰지 않는다. 약한 보조: 기술 통계에서 보정된 LGBM 팔 vs 보정된 LR_C(게이트 아님).

## 8. 사전 등록 판정 규칙 (팔별, 자기 보정 전 대비)

주 판정은 **geo2 4개 scheme(seed 42/101/202/303) × 5 Fold = 20 바깥 Fold** 전체 기준.

- P1: Fold Macro F1 평균 차이(보정 − 보정 전) ≥ **+0.003**
- P2: 보정이 더 높은 Fold ≥ **14/20**
- P3: scheme(seed)별 평균 차이가 양수인 seed ≥ **3/4**
- G1: 풀링 3등급 재현율 증가 ≥ **+0.03**
- G2: 풀링 3→1 오류 수가 보정 전의 **1.25배 이하**
- G3: 풀링 1등급 재현율 감소 ≤ **0.03**
- G4(stratified 감시): `strat_s42` 5 Fold 평균 차이 ≥ **−0.003** (낙관적 Fold이므로 "해롭지 않음" 확인용으로만 사용)
- **팔 후보** = P1 ∧ P2 ∧ P3 ∧ G1 ∧ G2 ∧ G3 ∧ G4. **공통 규칙 후보** = 3개 팔 모두 후보.
- geo1은 기술 보고만(게이트 없음). 모든 수치는 규칙 충족 여부와 무관하게 보고하고, 유의성·인과 주장은 하지 않는다.
- 후보 판정이 나와도 홀드아웃 평가·최종 모델 변경은 없다(사용자의 명시적 결정 필요). 규칙을 채택하려면 `b`를 전체 개발 데이터 내부 OOF로 한 번 더 고정하는 별도 사전 등록이 필요하다.

## 9. 실행 순서 (학습 전 필수)

1. 메인 세션이 이 문서·설정·코드·테스트를 커밋(이후 작업 트리 깨끗해야 함).
2. `python -m experiments.grade3_decision_rule.run prepare --data ../../earthquake/data --output <T15 run> --code-commit <HEAD 40자>`: 모든 manifest/OOF SHA·정렬·`argmax==y_pred`·Group 게이트·내부 Fold 게이트 검증 후 설계 봉인(학습 0회).
3. `fit-all`(주 scheme 먼저, `--schemes`로 부분 실행 가능) → `finalize`(`analysis.json`) → 독립 감사 → 보고서.

## 10. 금지·경계

홀드아웃(`earthquake/data/final_holdout/`) 읽기 금지(`guard()`가 경로를 차단하고 테스트가 확인), 기존 브랜치·기존 비공개 결과 변경 금지(재사용은 읽기 전용), 원시 데이터·모델·OOF·행별 예측·manifest Git 업로드 금지, `preprocess.py` 변경 없음, 커밋·push는 메인 세션만, 보고서·집계 JSON은 독립 감사 전 커밋 금지. 모호하거나 막히면 설계를 바꾸지 않고 중단 후 보고.
