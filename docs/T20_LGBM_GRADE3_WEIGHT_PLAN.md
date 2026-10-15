# T20 — LGBM-O 3등급 학습 가중치 사전등록

- 실험 ID: `T20_lgbm_grade3_weight_v1`, 작업 브랜치: `peter`.
- 코드: `experiments/lgbm_grade3_weight/`, 전용 검사: `tests/test_lgbm_grade3_weight.py`.
- 새 비공개 출력: `local_reproduction/T20_20261015_lgbm_grade3_weight/run/`.
- 현재 단계: 구현·합성 검증. 컨트롤 타워의 검토·커밋 및 실행 관리자 승인 전에는 `prepare`와 실자료 학습을 실행하지 않는다.

## 가설과 단일 변경

검증된 T8/T13의 **LGBM-O + C 전처리**를 고정하고, 실제 등급 1/2/3 학습 행에 `sample_weight=1/1/1.5`를 적용한다. 가중치는 원래 등급에서 만들며 LightGBM 학습 라벨만 `damage_grade-1`로 바꾼다. 내부 클래스는 `[0,1,2]`, 확률 열은 실제 등급 `[1,2,3]` 순서다. `class_weight=None`을 유지한다. 1.5는 최적값이라는 뜻이 없는 사전 고정 단일 가설이다.

모델은 300 rounds, 31 leaves, 학습률 0.05, 2 threads 및 나머지 T8/T13 파라미터를 그대로 쓴다. C 전처리는 각 학습 Fold 안에서만 fit하며 원본 38개 입력·특수값·희소 CSR을 유지한다. 검증 행과 지표에는 가중치를 주지 않는다. 예측은 원래 확률의 `argmax+1`이며 보정, early stopping, sampling, 하이퍼파라미터 탐색이 없다. 가중치 sweep, T17의 b, T18 선택 설정을 결합하지 않는다.

## 분할과 기준선

geo2 Group 분할 seed **42/101/202/303**, seed별 5Fold로 총 20Fold다. 그룹은 `(geo_level_1_id, geo_level_2_id)`다. 기존 manifest를 새로 만들지 않고 읽는다. 무가중치 LGBM-O 기준선은 seed42의 T8 OOF 및 seed101/202/303의 T13 OOF를 재사용한다. config에 기존 T15/T18과 동일한 manifest·OOF 경로와 전체 SHA-256을 고정하고, 바이트 SHA·ID·정답·Fold·확률 argmax 정합을 검증한다. 기준선 신규 학습은 0회다.

같은 개발 행을 네 번 재분할해 평가한다. 20개 Fold나 풀링 예측 행을 독립 표본 수로 해석하지 않는다. seed42와 개발 데이터는 앞선 연구에서도 반복 사용됐다. 홀드아웃에 접근하지 않으며 최종 모델을 변경하지 않는다.

## 예산과 사전판정

신규 학습은 단일 가중치 팔 × 4 seed × 5Fold = **20회, 최대 20회 시도**다. 순차 Fold, 자동 재시도 0회다. 실패·중단된 Fold는 원본과 실패 로그를 보존하고 중단한다. 같은 실험에서 실패 Fold를 다시 학습할 수 없다. 완료 Fold는 모델·확률·지표·백업 SHA를 확인하고 건너뛴다.

다음 다섯 항목을 모두 충족할 때만 `supports_fixed_hypothesis=true`로 기록한다.

1. 동일 20Fold의 무가중치 기준선 대비 Fold Macro F1 평균 차이 ≥ **+0.003**.
2. 엄격히 F1이 증가한 Fold ≥ **14/20**. 동률은 승리가 아니다.
3. seed별 5Fold 평균 F1 차이가 엄격히 양수인 seed ≥ **3/4**.
4. 반복 OOF 풀링 실제 3등급 Recall 차이 ≥ **0**.
5. 반복 OOF 풀링 3→1 오류율 차이 ≤ **0**. 분모는 실제 3등급 support다.

이 규칙은 개발 CV상의 고정 가설 지지 기준이다. 통계적 유의성, 실무 허용, 안전성 보장 또는 최종 모델 선정 기준으로 해석하지 않는다. 충족 여부와 관계없이 모든 seed/Fold를 보고하며 사후 기준 완화는 없다.

## 저장·봉인·감사

`prepare`는 학습 없이 소스 전이 import 폐쇄(기존 선언 SOURCE와 로컬 Python import), 설정·이 계획·테스트, 동결 환경, 개발 데이터 계약, 기존 manifest·OOF 참조를 SHA로 결속한다. 모든 봉인 대상은 제공한 전체 커밋 SHA의 깨끗한 추적 파일이어야 한다. 재개 전 봉인과 백업을 재검증한다. T16/T18 등 기존 봉인 파일은 수정하지 않는다.

각 Fold는 모델 native text, 기존 C 전처리 joblib, 검증 ID·정답·확률·예측, 지표·학습 시간·학습 가중치 수, 학습/검증 CSR의 shape·dtype 및 data/indices/indptr SHA를 저장한다. 저장 직후 모델 재로딩 확률 최대 절대차 ≤ `1e-12`, CSV round-trip 완전 일치, 원본·별도 backup SHA를 확인한 뒤 원장 상태를 complete로 바꾼다. 기존 경로를 덮어쓰지 않는다. `RUNNING.lock`은 배타적 생성하며 기존 잠금을 임의 해제하지 않는다.

전처리 joblib의 `experiments.age995.run.Age995Treatment`는 로딩 시 실행기를 간접 import한다. 독립 감사는 일반 joblib.load를 사용하지 않고 독립 C 전처리 재구성 → CSR 배열 SHA 비교 → native Booster 재추론으로 진행한다. 이는 분류기 재학습이 아니다. 독립 감사는 별도 담당자가 실행기 import 없이 수행하며 PASS 전 공개 성능 보고를 확정하지 않는다.

집계는 Fold 평균/SD, seed별 pooled 및 전체 반복 OOF, 등급별 P/R/F1·support·혼동행렬·정확도, 실제3등급 Recall, 3→1/3→2와 2→3 오류 및 실제등급 분모, age995/비995를 남긴다. 원시 자료·모델·OOF·manifest는 Git에 올리지 않는다.

학습 가중치는 최적화 목적과 출력 확률을 바꿀 수 있다. 정규화된 확률 합이 1이어도 원래 모집단의 보정된 사후확률임을 보장하지 않는다. 이번 연구는 확률 보정·확률 품질 개선을 입증하지 않으며, 확률 기반 의사결정은 별도 검증 대상이다. T15의 사후 결정 규칙은 별개 기술 참고이며 개선량을 합치지 않는다.

## 실행 진입점

동결 Python: `../../local_reproduction/20261014_01/venv/Scripts/python.exe`. 설치·업그레이드 금지. `TEMP/TMP`는 T20 비공개 폴더, `PYTHONIOENCODING=utf-8`로 지정한다. 아래는 검토 가능한 명령 예시이며 실제 실행 승인을 대신하지 않는다.

```powershell
<동결 Python> -m experiments.lgbm_grade3_weight.run prepare --data ../../earthquake/data --output ../../local_reproduction/T20_20261015_lgbm_grade3_weight/run --code-commit <전체 SHA>
<동결 Python> -m experiments.lgbm_grade3_weight.run fit-all --data ../../earthquake/data --output ../../local_reproduction/T20_20261015_lgbm_grade3_weight/run --schemes geo2_s42 --folds 0
<동결 Python> -m experiments.lgbm_grade3_weight.run status --output ../../local_reproduction/T20_20261015_lgbm_grade3_weight/run
```

실행 관리자가 첫 단위 종료·비용과 자원 여유를 확인한 뒤 `fit-all`의 두 필터를 제거해 나머지를 실행한다. 완료20·프로세스 종료·잠금 해제 후 같은 모듈 `finalize`를 실행하고 독립 감사한다. 자원 압박 시 다음 단위 시작을 보류한다. 실자료 실행과 무거운 감사의 유일한 관리자는 컨트롤 타워가 지정한 Astra 실행 관리자다.
