# T14 — 팀원 탐색 파이프라인(TE + ROS + RF) 로컬 재현 사전 등록 계획

- 실험 ID: `T14_TE_ROS_RF_LOCAL_REPRO`
- 브랜치: `peter-exp/te-ros-rf-local-repro-v1` (기준 `af261c7` = T13 보고서 커밋)
- 코드: `experiments/te_ros_rf_repro/` (`run.py`, `config.json`), 원본 보존본: `experiments/te_ros_rf_exploratory/`, 테스트: `tests/test_te_ros_rf_repro.py`(합성 데이터만)
- 비공개 출력: `local_reproduction/T14_20261015_te_ros_rf_repro/` (Git 밖, 사전에 존재하지 않아야 함)
  - `preflight/`, `run/`(설계 봉인·분할 사본·원장·Fold 산출물·`backup/`), `logs/`
- 이 문서·설정·코드·테스트는 학습 전에 커밋되고, 그 **전체 커밋 SHA**로 `prepare`가 `design.json`/`design_seal.json`을 봉인한다.
- 확정(2026-10-15, 사용자가 메인 세션 추천안에 위임): Stage A 허용 오차와 Stage B 판정 기준값을 초안대로 확정한다. Stage B 내부 TE Fold는 그룹 인지 방식만 쓴다. 원본의 무작위 내부 Fold는 T10에서 분포 쏠림이 확인돼 추가하지 않는다. 정식 preflight는 `local_reproduction/T14_20261015_te_ros_rf_repro/preflight`에서 다시 실행한다. T15 학습이 끝난 뒤 순차로 학습한다(동시 학습 없음).

## 0. 이 실험의 성격 (먼저 읽기)

- **재현(reproduction)이다.** 웹 세션에서 실행해 `origin/peter-exp/te-ros-rf-exploratory-v1`(`0ecd351`)에 올린 탐색 결과(팀원 아이디어 기반)를 로컬에서 같은 절차로 다시 학습해 그 값이 나오는지, 그리고 같은 파이프라인이 지역 Group 분할에서 어떻게 동작하는지 **기록**한다.
- 원본은 집계 혼동행렬만 있고 모델·행 단위 OOF가 없어 **미감사**다(요약 JSON `audit_scope`). T14는 그 숫자를 감사 가능한 형태(Fold별 모델·확률·SHA 원장)로 다시 만든다.
- **요인 분리를 주장하지 않는다.** 원본은 TE, ROS, 모델 종류(RF), 연령 처리, 원시 geo ID 유지를 동시에 바꿨다(요약 JSON `limitations`). 어느 요소가 이득·손실을 만드는지는 이 실험으로 알 수 없고, **후속 실험(요인 하나씩 변경)** 에서 다룬다. 결과 보고서에서 "TE 덕", "ROS 덕", "RF 덕"이라고 쓰지 않는다.
- 새 방법·튜닝·임계값 조정·최종 모델 변경·홀드아웃 평가 없음. 기준선은 LR-C 유지가 기본이다.

## 1. 원본 파이프라인 (그대로 보존, 해시 고정)

보존 위치 `experiments/te_ros_rf_exploratory/`(원본 파일을 `git show 0ecd351:<경로>`로 바이트 그대로 가져옴, LF 저장·CR 0개):

| 파일 | SHA-256 (요약 JSON `source_sha256`과 일치 확인) |
|---|---|
| `evaluate_te_ros.py` | `435975ae64f5ae034764f8e85f7f718b81532da640ff44d53fe1424d325c7e61` |
| `validate_rf_5fold.py` | `4dba1f1aec28b13d8a509cf94499ce5b71ca9d53f9225b31836062a314b0d585` |
| `README.md` | `99323ba40e8f858f15ed75c7c3bff15945478ff2b2387a89ee3c32378dfe49c5`(Git 저장본; 요약 JSON의 `README_exploratory.md`와는 다른 파일) |

- 요약 JSON(`reports/te_ros_rf_exploratory_summary.json`, Git blob SHA-256 `66f8d74f90beb06cc89b5dcfcd2050b55585537af1dfefaed43edc269ac7670d`)의 값이 `config.json` `recorded_reference`에 복사돼 있다(Fold별 혼동행렬 포함). 요약 JSON의 `source_sha256`에 있는 `te_ros_fold0_results.json`, `rf_5fold_result.json`, `README_exploratory.md`는 ZIP 멤버로 이 저장소에 없다.
- 재현하는 절차: geo1/geo2/geo3 각각 클래스 1·3 비율의 OOF 타깃 인코딩(내부 5-Fold, m=20, 내부 학습 Fold 평균을 prior로) + 지역별 빈도 3열(학습 Fold 기준, 검증은 학습 빈도 매핑·미관측 0) → TE 이후 **바깥 학습 Fold에만** `RandomOverSampler(random_state=0)` → 원시 geo ID(숫자)와 원시 `age`(995 포함) 유지, 문자 범주 8열 one-hot, **모델 열 77개** → `RandomForestClassifier(n_jobs=4, random_state=0)`(나머지는 sklearn 1.8.0 기본값; `config.json`에 get_params 스냅샷 고정).
- 검증 인코딩은 바깥 학습 Fold 전체 통계, 검증 prior는 바깥 학습 Fold 클래스 비율. 학습 행 인코딩의 내부 Fold prior는 내부 학습 Fold 값(원본이 "내부 OOF prior 누수 제거"로 명시한 변경).
- 원본이 Fold 0만 `evaluate_te_ros.py`로, Fold 1-4는 `validate_rf_5fold.py`로 학습했고 두 스크립트의 인코딩 함수는 같다. T14는 **5개 Fold 모두 새로 학습**한다(원본의 Fold 0 재사용 없음).

### 원본 코드와의 의도된 차이 (전부 기록)

1. 데이터 읽기: 원본은 `development.csv`를 직접 읽는다. T14는 `load_development()`(`experiments.geo_group_cv.run.load`, 데이터 계약 SHA 검증 포함)로 읽고 `building_id` 정렬·저장된 fold만 쓴다. 정규화 결과(정수/문자열)에서 원본 프레임(geo 정수, 문자 열 category, 나머지 정수)을 복원하며, 열 순서는 CSV와 같다. 이진 열이 `uint8`인 점만 다르고 `float32` 변환 후 값은 같다.
2. 범주 어휘: 원본처럼 개발 세트 전체에서 관측된 값으로 `category`를 만든다(라벨과 무관한 스키마, 통계 fit 아님).
3. 인코더 함수 `target_enc`는 원본 본문 그대로에 훅 두 개만 추가했다: `splits`(내부 분할 주입; 기본은 원본의 `StratifiedKFold(5, shuffle, 0)`)와 `return_state`(재로딩 검증용 통계 보관). 원본 함수를 AST로 추출해 **합성 데이터 테스트와 실데이터 Fold 0 사전 점검에서 출력 동일성**을 확인한다.
4. 예측: 원본 `predict` 대신 `predict_proba`의 argmax를 쓴다(sklearn RF `predict`와 동일 규칙). 확률을 저장하기 위함이다.
5. 지역 Group 분할(Stage B)의 내부 인코딩 Fold를 T11 방식으로 바꾼다(아래 3절). Stage A는 원본 그대로.
6. Fold 0 단일 5모델 비교(LR/DT/XGB/LGBM)는 재현 범위 밖이다. RF만 다룬다.

## 2. 입력 (학습 전 `prepare`가 검증, 사전 점검에서 확인한 값)

| 입력 | 경로 / 값 |
|---|---|
| `development.csv` | `earthquake/data/development.csv`, SHA `8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1` |
| `split_manifest.csv` | `earthquake/data/split_manifest.csv`, SHA `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721` |
| geo2 manifest seed 42 | `local_reproduction/T6R_20261014_local/run/geo2/manifest.csv`, SHA `36a40ddf1153b7a25bf36d5573b26b4dc9a0d67d240869c2e5cc1d85c1f89d64`(T5 사본과 바이트 동일 확인) |
| geo2 manifest seed 101 | `local_reproduction/T13_20261014_geo2_te_confirm/run/manifests/seed_101/manifest.csv`, SHA `7877b21b59433dea9c4a22ddd8fc1112a3440180fc4e2ab9a73aa0a5b56af1f4` |
| geo2 manifest seed 202 | `…/manifests/seed_202/manifest.csv`, SHA `9b3a48054743a95b83d85d95ac05354456d441fe4ac6c8aa189434d3877914c7` |
| geo2 manifest seed 303 | `…/manifests/seed_303/manifest.csv`, SHA `26f42c0b2b3ea40f736d82d178b864ce2d4a25d1eeb2d924c65a177254b8fafd` |

(manifest의 전체 SHA는 `config.json` `manifests`에도 있고 `prepare`가 일치를 요구한다.)

## 3. 단계

### Stage A — 고정 stratified 분할 충실 재현 (5 fit)

- `split_manifest.csv`의 `cv_fold` 0-4(`load_development()`가 돌려주는 저장 fold). 내부 TE Fold는 원본 `StratifiedKFold(5, shuffle=True, random_state=0)` 그대로.
- Fold별 혼동행렬 [참×예측, 등급 1·2·3]을 원본 기록과 비교한다. 사전 점검(학습 없음)에서 확인한 사항: Fold 0 학습 166,784 / 검증 41,696 / ROS 후 284,655행(클래스당 94,885), 모델 열 77개가 기록과 일치하고 인코딩 행렬이 원본 함수 출력과 **비트 동일**하다.

**재현 판정 (사전 고정, `rules.stage_a`)**

- `EXACT`: 5개 Fold의 혼동행렬이 모든 셀에서 기록과 동일.
- `WITHIN_TOLERANCE`: EXACT가 아니고 아래를 **모두** 만족.
  - Fold별 |Macro F1 − 기록| ≤ 0.002
  - |5Fold 평균 − 0.7004271| ≤ 0.001
  - 풀링 3등급 Recall |차이| ≤ 0.005 (기록 0.67302)
  - 풀링 3→1 건수 상대 차이 ≤ 5% (기록 744)
  - Fold별 혼동행렬 셀 최대 |차이| ≤ 해당 Fold 검증 행 수의 0.5%
- `FAIL`: 그 외. **Stage B를 실행하지 않고** 중단·보고한다(파이프라인을 재현하지 못한 상태에서 지역 분할 결과를 원본과 연결해 해석하지 않기 위해).
- 동일 환경(sklearn 1.8.0, pandas 2.2.3, numpy 2.3.5)에서 RF는 `random_state=0`으로 결정적이므로 EXACT를 기대하지만, 확률 합산 순서·CPU 차이를 감안해 허용 오차를 둔다. 허용 오차는 학습 전에 고정하며 결과를 본 뒤 바꾸지 않는다.
- 판정은 `finalize-a`가 `stage_a_result.json`으로 기록하고, 이 파일이 `EXACT`/`WITHIN_TOLERANCE`가 아니면 `fit-all --stage B`가 거부한다.

### Stage B — geo2 Group 분할 4개 (20 fit)

- 분할: seed 42(T5 geo2) + seed 101/202/303(T13). 모두 기존 감사된 manifest 사본(바이트 SHA 검증)이며, `prepare`가 T5 게이트(그룹·ID 서로소, 클래스별 ≥ 30행, 검증 크기 1%~50%)를 다시 확인한다.
- **내부 TE Fold 선택(문서화된 결정)**: Group 분할에서는 T11과 같이 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, 그룹 = (geo1, geo2) 합성(`experiments.geo_te_group_inner.encoder.inner_group_fold_ids`)을 쓴다. 이유: 원본의 무작위 내부 Fold는 학습 행의 인코딩을 같은 지역의 다른 행으로 계산해 "관측된 지역" 분포를 만들지만 Group 검증 행은 미관측 지역이라 분포가 어긋난다(T10에서 예측이 3등급으로 쏠린 원인; T11이 해소). 선택은 사전 점검 전에 정했고 결과를 본 뒤 바꾸지 않는다.
- **알려진 한계(사전 점검으로 관측, 판정에 영향 없음, 결과 해석 시 반드시 명시)**
  - geo2 Group 분할에서 검증 행의 geo2·geo3 ID는 학습 Fold에 **100% 없다**(관측 비율 0.0, 지역 빈도 0, TE는 prior). geo1은 100% 관측.
  - 학습 행의 geo2·geo3 TE는 그룹 인지 내부 Fold 구조상 항상 내부 학습 Fold의 prior만 가지므로 정보가 없다(geo3는 geo2에 포함). 지역 빈도 열은 학습에서는 양수(예: geo2 평균 약 416), 검증에서는 0이라 **학습–검증 불일치**가 남는다. 원시 geo ID 숫자 열도 검증에서는 학습에 없는 값이다.
  - 따라서 Stage B는 "TE가 지역 일반화에 기여하는가"를 판정하지 않는다. 원본 파이프라인 전체가 미관측 지역에서 어떻게 동작하는지 기록한다. 각 Fold의 TE·빈도 학습/검증 분포(`geo_diagnostics`)와 원시 geo ID/TE/빈도/연령의 RF 중요도 비중을 저장해 보고한다.
- Stage A와 동일 RF·ROS·인코딩 설정. 열 수 77은 모든 분할에서 사전 점검 확인.

## 4. 비교군 (재학습 없음, 기존 감사된 OOF 재사용)

`prepare`가 경로 존재·SHA·Fold 열이 분할과 일치·라벨 일치를 확인한다(전체 경로·SHA는 `config.json` `comparators`).

| 분할 | LR-C | LGBM-O (one-hot) | LGBM_TE |
|---|---|---|---|
| stratified | `local_reproduction/20261014_01/private_original/age995_abc_v1/C/lr/oof_predictions.csv.gz` `de43f997…2cbc` | `…/private_t4_original/private/model_families_v1/LGBM_C_v1/oof_predictions.csv.gz` `0bc63eb5…30cf` | `T10_20261014_geo_te/run/stratified/LGBM_TE/oof.csv.gz` `3c0d2cd8…d859`(T10 계층적 TE, 참고용) |
| geo2 seed 42 | `T6S_20261014_c_group/t5_private/outputs/geo_group_cv_v1/geo2/C/oof.csv.gz` `9e58f72b…00b4` | `T8_20261014_native_cat/run/geo2/O/oof.csv.gz` `ba282656…a211c` | `T11_20261014_te_group_inner/run/geo2/LGBM_TE_G/oof.csv.gz` `34f23ad6…039b` |
| geo2 seed 101 | `T13_20261014_geo2_te_confirm/run/s101/LR_C/oof.csv.gz` `486f7975…` | `…/s101/LGBM_O/oof.csv.gz` `bd2f5f17…` | `…/s101/LGBM_TE_G/oof.csv.gz` `5bc7df13…` |
| geo2 seed 202 | `…/s202/LR_C/oof.csv.gz` `1c8ef79e…` | `…/s202/LGBM_O/oof.csv.gz` `45d0d0bb…` | `…/s202/LGBM_TE_G/oof.csv.gz` `2fa8f729…` |
| geo2 seed 303 | `…/s303/LR_C/oof.csv.gz` `7dc09af6…` | `…/s303/LGBM_O/oof.csv.gz` `a3666891…` | `…/s303/LGBM_TE_G/oof.csv.gz` `0b8ff718…` |

(SHA 앞 8자리와 전체값은 사전 점검에서 계산·확인한 것이고 `config.json`에 전체값이 있다. 비교는 설계가 다른 모델 간 기술 비교이며, 특히 stratified 비교는 원본이 명시한 대로 `comparison_is_controlled = false`.)

## 5. 사전 등록 판정 규칙

**Stage A**: 위 3절의 `EXACT` / `WITHIN_TOLERANCE` / `FAIL`. 그 외 Stage A 수치(Macro F1 평균·SD, 클래스별 P/R/F1, 3등급 Recall, 3→1·3→2)는 기록·보고하며, 원본 0.700427 초과를 배포 성능으로 해석하지 않는다.

**Stage B** (20 Fold = 4 분할 × 5 Fold, Fold 단위 ΔMacro F1 = RF − 비교군, labels=[1,2,3]):

- **B1 (vs LR-C)**: 평균 Δ > 0 **AND** RF 승 ≥ 16/20 **AND** 풀링 3등급 Recall 차이 ≥ −0.01 **AND** 평균 Δ > 0인 분할 ≥ 3/4.
- **B2 (vs one-hot LGBM-O)**: 평균 Δ ≥ +0.010 **AND** RF 승 ≥ 16/20 **AND** 풀링 3등급 Recall 차이 ≥ −0.01 **AND** 평균 Δ > 0인 분할 ≥ 3/4.
- 라벨: `B1_AND_B2` / `B1_ONLY` / `B2_ONLY` / `NEITHER`. 이 라벨은 "이 RF 파이프라인이 미관측 지역에서도 해당 비교군을 앞서는가"의 기술적 요약일 뿐, 어느 요인 때문인지, 최종 모델로 적합한지는 말하지 않는다.
- 임계값 없이 보고: vs LGBM_TE_G(T11/T13 모델) 평균 Δ·승수, 분할별 평균 Δ, 3등급 Recall, 3→1·3→2 건수와 3등급당 비율(비교군과 나란히; 원본은 stratified에서 3→1 = 744로 LR-C 452보다 많았음), 정확도, 예측 비율, 같은 행 turnover(`experiments/geo_ablation/audit.py turnover()`), age 995 및 geo1 26/20/13 프로파일, `geo_diagnostics`, 중요도 비중.
- 유의성·인과 주장 없음. 모든 수치를 규칙 충족 여부와 무관하게 보고한다. 홀드아웃 평가·최종 모델 변경은 사용자의 별도 결정 사항이다.

## 6. 학습 횟수·시간·병렬

- **신규 RF 학습 25회 = Stage A 5 + Stage B 20.** 비교군 재학습 0, 자동 재시도 0, Fold 순차, 외부 병렬 없음.
- **병렬은 한 단계만**: RF `n_jobs=4`. TE·ROS·평가는 단일 스레드. 희소 행렬을 쓰지 않으며(밀집 float32 약 285k×77) `.toarray()` 없음.
- 사전 점검(분류기 학습 없음) 실측: TE 인코딩 0.4–0.8 s, ROS 0.2 s / Fold.
- **RF 학습 시간은 아직 측정하지 않았다(미측정).** 100그루 완전 성장 트리, 약 28.5만 × 77, 4스레드 기준으로 Fold당 대략 3–8분 + 모델 저장(압축)·재로딩 검증 1–3분이라는 **거친 추정**일 뿐이며, 전체 약 2–5시간으로 예상한다. Stage A 첫 Fold의 실측 시간으로 이 추정을 고쳐 보고한다. 추정이 틀려도 설계·판정은 바뀌지 않는다.
- 디스크: RF joblib(압축 3)은 Fold당 수백 MB~1 GB 이상일 수 있어 Fold당 원본+`backup/` 2벌이다. `prepare`는 여유 ≥ 60 GB, 각 Fold 시작 시 ≥ 20 GB를 요구한다(현재 C: 여유 약 727 GB).

## 7. 체크포인트·감사 가능성 (T13 실행기와 같은 방식)

- Fold 완료 즉시 저장: `fold_{f}_model.joblib`(RF), `fold_{f}_encoder_state.joblib`(전체 학습 통계: 열 이름·범주·지역별 TE 매핑·prior·빈도), `fold_{f}_pred.csv.gz`(확률, 17자리 round-trip), `fold_{f}.json`(지표·혼동행렬·진단·시간·파라미터). SHA-256을 `ledger.json`에 기록하고 `backup/`에 복사한다.
- **재로딩 검증**: 저장한 모델과 인코더 상태를 다시 읽어 검증 행을 원시 특성에서 재인코딩 → `predict_proba`가 저장된 확률과 atol 1e-12 이내, 라벨 100% 일치(멀티스레드 합산 순서 차이만 허용). CSV round-trip은 정확히 같아야 한다.
- 완료 Fold는 해시를 검증만 하고 재학습하지 않는다. 중단 후 재개: `running` 상태 항목의 미완 산출물은 `.incomplete_attempt<n>`으로 치우고 수동 재개한다. 학습 예산(25회)·미등록 Fold·Stage B 선행 조건은 코드가 강제한다. `RUNNING.lock`으로 중복 실행 방지.
- 설계 봉인(`design.json`/`design_seal.json`): 설정·소스 SHA·환경·분할 SHA·비교군 SHA·사전 점검 결과를 학습 전에 고정한다.

## 8. 실행 절차 (학습 시작은 이 문서·코드 커밋 및 사용자 승인 후)

1. (완료 가능, 학습 없음) `python -m experiments.te_ros_rf_repro.run preflight --data ../../earthquake/data --destination <out>/preflight` → `status = PASS` 필수.
2. `prepare --output <out>/run --code-commit <전체 SHA> --preflight <out>/preflight/preflight.json` (소스는 커밋·clean 이어야 함).
3. `fit-all --stage A` → `finalize-a`(재현 판정) → EXACT/WITHIN_TOLERANCE이면 `fit-all --stage B` → `finalize`(분석 JSON).
4. 독립 감사(메인 세션) 후에만 보고서·집계 JSON을 커밋한다.

## 9. 금지·경계

홀드아웃·`final_holdout` 읽기 금지(경로 가드와 테스트 포함), 기존 브랜치·기존 비공개 결과 변경 금지, 원시 데이터·모델·OOF·manifest·행별 예측 Git 업로드 금지, push 금지, 보고서·집계 JSON은 메인 세션 독립 감사 전 커밋 금지. 모호하거나 막히면 설계를 바꾸지 않고 중단 후 보고. 후속 요인 분리(TE 단독, ROS 단독, RF 단독, age995 처리 등)는 이 실험이 끝난 뒤 별도 사전 등록한다.
