# T16 — RF 기준 파이프라인의 요인별 지역 일반화 비교 사전 등록

- 실험 ID: `T16_RF_FACTORS`, 브랜치: `peter-exp/t16-rf-factors`.
- 코드: `experiments/rf_factors/`, 합성 계약 테스트: `tests/test_rf_factors.py`.
- 비공개 새 경로: `local_reproduction/T16_20261015_rf_factors/` 아래 `preflight/`, `run/`, `audit/`.
- 기존 T14 결과, 데이터, 모델, OOF, manifest는 읽기 전용이며 공개 Git에 추가하지 않는다. 홀드아웃은 접근하지 않는다.
- 사전 등록 코드 커밋 → 학습 없는 `preflight` → `prepare` 봉인 → 단일 실행권을 받은 실행자만 학습 → 독립 감사 → 집계 보고서 순서다.
- 사용자 승인 범위는 T14 요인 분리 실행이다. 기준선 LR-C, 최종 모델, 홀드아웃 평가 계획은 변경하지 않는다.

## 1. 비교 질문과 고정 기준

T14는 geo2 4분할에서 RF 파이프라인 전체가 LR-C와 LGBM보다 낮았다. 어느 요소의 변경이 그 차이를 줄이는지 기준 arm에 대해 하나씩 비교한다. 새 방법 탐색이나 최적 조합 선택은 아니다. 상호작용을 추정하지 않으며, 개별 팔의 이득을 합산하거나 조합의 성능으로 해석하지 않는다.

기준 arm `T14_RF_TE_ROS_A`는 감사 PASS인 T14 `run_v2`의 geo2 OOF를 재사용한다. 설계 SHA는 `965f8741dd48ea0b277a9a512b551fd05ac1f365bd9a4d11fd9769688db09a74`, 사전 등록 커밋은 `813fab309b42fd4481316aa99aa9b71959b47c9a`다. 네 OOF와 manifest의 전체 SHA는 `config.json`에 고정한다. `preflight`, `prepare`, 학습 시작, 최종 집계가 기준 설계/소스/SHA/ID/라벨/Fold 정렬을 다시 검증한다. 기준 RF 재학습은 0회다.

기준의 나머지 설정은 T14와 같다. 원시 입력 38개 보존, raw geo 숫자 3열 유지, 문자 8열의 기존 범주 스키마 one-hot 유지, geo1/2/3별 등급1·3 TE 및 빈도, m=20, 내부 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`의 (geo1,geo2) 그룹, outer-train에만 ROS(random_state=0), age995 A(원시값), RF 100 trees/random_state=0/n_jobs=4와 나머지 sklearn 1.8.0 기본값이다. 예측은 단일 스레드로 하며 `argmax+1`로 라벨을 만든다.

## 2. 사전 고정 팔 — 각각 기준과만 비교

| arm | 기준 대비 바꾸는 요인 | 그대로 고정하는 요소 |
|---|---|---|
| `ONEHOT` | geo TE 6열을 outer-train에서 학습한 geo one-hot 블록으로 교체 | raw geo 숫자, geo 빈도 3열, ROS, age A, RF |
| `NO_ROS` | ROS를 끈다 | 모든 전처리 값/열, age A, RF |
| `DROP_GEO23_TE` | geo2·geo3의 TE 4열만 제거한다 | geo1 TE, 모든 raw geo 숫자/빈도, ROS, age A, RF |
| `AGE_C` | age995에 C 처리: outer-train non995(0 포함) 중앙값 대체 + `age_is_995` 표시 | geo 인코딩/빈도, ROS, RF |

`ONEHOT`은 **TE 블록 대체 효과**다. 전체 geo 범주 처리 파이프라인이나 기존 LGBM one-hot과의 동일 조건 비교라고 쓰지 않는다. 알려지지 않은 geo 범주는 one-hot 0이며 학습 범주만 어휘에 넣는다. 희소 CSR을 유지하며 전체 `.toarray()` 변환은 없다. 원시 geo 숫자를 함께 남기는 것은 기준 대비 변경을 TE 블록 하나에 제한하기 위해서다.

`DROP_GEO23_TE`는 빈도 제거 효과를 함께 섞지 않는다. C의 중앙값과 표시 변수는 사전에 정의된 연령 처리 정책 한 묶음이며 각각의 효과를 분리하지 않는다. 인코딩/특성 수가 달라지는 팔은 RF의 고정 `max_features='sqrt'`에 따라 후보 특성 수도 달라진다. 따라서 고정 RF 알고리즘에 해당 표현 변경을 적용한 조건부 비교이며, 차원이 통제된 순수 정보량 효과라고 주장하지 않는다.

내부 TE는 각 내부 학습 Fold의 라벨만 쓰고 outer validation은 전체 outer-train 통계로 변환한다. ROS는 TE 뒤 outer-train에만 적용한다. 연령 중앙값·one-hot 어휘도 outer-train에서만 학습한다. 검증 라벨은 인코딩 함수의 인자가 아니다. 범주 스키마 외의 통계 fit에 전체 개발 데이터를 사용하지 않는다.

## 3. 분할·예산·중단 처리

geo2 seed 42/101/202/303의 기존 manifest를 그대로 사용한다. 각 5 Fold, 각 Fold 4팔로 신규 완료 fit은 80개다. 학습은 팔/Fold/분할 전체에서 직렬이고 RF 안에서만 4스레드로 병렬화한다. 튜닝·early stopping·결정 규칙 변경·자동 재시도는 모두 없다.

인프라 중단만 `--retry-interrupted --retry-reason "사유"`로 수동 재시도할 수 있다. 동일 봉인 설계로 Fold당 최대 1회, 전체 최대 4회 추가를 허용하여 전체 **84 fit attempts**를 상한으로 한다. 완료 Fold는 SHA와 백업을 검증하고 재학습하지 않는다. 실패 원장·로그·이벤트와 부분 파일은 `.incomplete_attempt1`로 보존한다. 모델 성능을 본 뒤 파라미터를 바꾸는 용도가 아니며, 알고리즘/데이터 결함은 중단 후 새 개정 설계로 다룬다. 프로세스 사망 후 남은 `RUNNING.lock`은 실제 실행 주체가 없는지 컨트롤타워에서 확인한 뒤에만 제거한다.

T14의 25 fit 학습시간 합계는 243.84초였다. 같은 규모라면 신규 80 fit 학습만 약 13분이지만 희소 고차원 ONEHOT 비용과 저장/재로딩/백업 시간을 포함하면 15~45분을 잠정 예상한다. 이는 완료시간 보장이 아니며 첫 Fold 실제 비용으로 갱신한다. 준비 시 여유 디스크 60GiB, 각 Fold 전 15GiB를 요구한다. 신규 출력은 원본과 백업을 함께 두므로 저장 비용도 관찰한다.

## 4. 판정과 보고 범위

주지표는 labels=[1,2,3] Macro F1의 동일 Fold 간 차이다. 각 팔을 기준과 비교하여 20 Fold 평균 Δ≥0.003, 양의 Δ인 Fold≥16/20, 양의 분할 평균≥3/4, 풀링 3등급 Recall Δ≥−0.01을 모두 만족하면 `SUPPORTED_CONTRAST`로 표시한다. 그 외는 `NOT_SUPPORTED`다. 이 명칭은 지정 변경을 지지하는 기술 통계 기준이며 최종 모델 후보 승격이 아니다.

분할별/20 Fold Δ, 클래스별 Precision/Recall/F1, 혼동행렬, accuracy, 3→1/3→2 오류 및 예측 비중, 기준 대비 예측 전환, 시간·특성수·변수 중요도를 남긴다. 풀링은 같은 건물 4회가 포함되는 반복 분할 요약임을 명시한다. Fold/분할은 독립 관측이 아니며 p-value나 인과적 요인 효과를 주장하지 않는다. 4팔 중 최고 팔 선정이나 각 팔의 차이 비교는 사전등록된 주 판정 밖이다.

## 5. 저장·봉인·감사

각 Fold에서 모델·인코더 상태·17자리 확률 CSV·집계 JSON을 저장하고 SHA 원장, 단일 스레드 모델/상태 재로딩의 확률 비트 동일성, CSV round-trip, 별도 백업 SHA를 검증한 뒤 완료로 기록한다. 저장 실패 시 완료로 표시하지 않는다. 최종 OOF 조립도 중단 후 재개 가능하고 기존 값이 다르면 실패한다.

독립 감사는 실행기 import 없이 구현하여 분할/OOF/모델/TE·one-hot·age 통계와 변경 요인의 고정 여부를 재계산한다. 감사 전에는 보고서를 결과 확정으로 커밋하지 않는다. `preflight`는 모든 분할의 Fold0에서 4팔 전처리와 상태 재생성, T14 기준 행렬 비트 동일성을 확인하며 실제 분류기 학습은 0회다.

## 6. 실행 명령 (worktree 루트, 동결 Python 사용)

`P`는 `C:/Projects/02_Team_Project/total_earthquake/local_reproduction/20261014_01/venv/Scripts/python.exe`, `OUT`은 `../../local_reproduction/T16_20261015_rf_factors`다. Windows에서는 TEMP/TMP를 이 새 private 경로 아래 쓰기 가능한 별도 폴더로 두고 `PYTHONIOENCODING=utf-8`을 설정한다.

```text
P -m unittest discover -s tests -p test_rf_factors.py -v
P -m unittest discover -s tests -v
P -m experiments.rf_factors.run preflight --data ../../earthquake/data --destination OUT/preflight
P -m experiments.rf_factors.run prepare --data ../../earthquake/data --output OUT/run --code-commit <전체 사전등록 SHA> --preflight OUT/preflight/preflight.json
P -m experiments.rf_factors.run fit-all --data ../../earthquake/data --output OUT/run
P -m experiments.rf_factors.run status --output OUT/run
P -m experiments.rf_factors.run finalize --data ../../earthquake/data --output OUT/run
```

학습 명령은 컨트롤타워가 단일 실행권을 부여한 후에만 실행한다.
