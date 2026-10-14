# T6 실행 증거 회수 후 감사 인계

2026-10-14 KST. 현재 판정은 `BLOCKED_T6_PRIVATE_EVIDENCE_MISSING`이다. [회수 보고서](../../reports/GEO_ABLATION_RETRIEVAL_20261014.md)와 [집계 JSON](../../reports/geo_ablation_retrieval_20261014.json)을 먼저 읽는다.

## 1. 기존 런타임에서 받아야 할 비공개 증거

원래 실행자 또는 해당 컨테이너/로컬 에이전트의 보관 경로에서 다음을 **읽기 전용으로 목록화하고 묶는다.** 경로가 사라졌다는 사실만으로 학습을 재시도하지 않는다. 없는 파일은 missing으로 남기고 SHA를 만들지 않는다. ZIP은 비공개 전달하며 공개 Git에 올리지 않는다.

```text
outputs/geo_ablation_v1/
  design.json, design_seal.json, execution_interruption.json
  geo2/manifest.csv, geo1/manifest.csv
  stratified|geo2|geo1/R12|R1|R0/
    metadata.json, folds.json, fold_*.joblib
    oof.csv.gz, metrics.json (실제로 생성된 구성만)
  audit_available/ (audit.json, 집계·부분 OOF·hash 근거)
outputs/geo_ablation_unattempted_v1/ (실제로 존재하는 경우)
  supplement_design.json, supplement_seal.json
  progress.json, folds.json, fold_3.joblib, fold_4.joblib
  partial_oof.csv.gz, inspections.json, audit.json
```

학습·중단·보완·감사 stdout/stderr 및 시도 원장, 생성 시각과 원래 실행 환경 기록도 포함한다. ZIP 멤버 목록과 각 멤버의 SHA-256, ZIP 자체 SHA-256을 별도로 남긴다. 재압축 ZIP의 SHA가 새로 바뀌더라도 원래 멤버 바이트는 보존돼야 한다. design/seal을 결과 확인 후 새로 만들거나 과거 시각으로 소급하지 않는다.

현재 Library에 보이는 C 기준선 복구 ZIP은 T6 결과 ZIP이 아니다. T6 코드 ZIP과 공개 Git도 실행 원장을 대신하지 않는다. 원래/보완 폴더가 없는 경우 **어디까지 찾았는지와 실제 보관 위치/보관본 이름을 확인할 수 없는 이유**를 적는다. 이 문서는 다른 채팅에 자동으로 메시지를 보내는 승인이 아니다.

## 2. 회수 후 학습 없는 사전 검사

1. 최신 T6 원격 HEAD·동시 변경을 확인하고 새 복사본을 사용한다. 다른 브랜치와 기존 로컬 출력은 보존한다.
2. 신뢰 가능한 프로젝트 생성 자료인지 확인하고 모델을 로드하기 **전에** 패키지 SHA·design/seal·metadata·folds의 모델/OOF SHA를 대조한다. 경로 탈출과 중복 ZIP 멤버도 거부한다.
3. 원래 학습 코드 `87b8a1d5d0e7918794f75bbd7e9a5e8418295574`와 실제 봉인 `source_sha256`의 바이트가 일치하는지 확인한다. 공개 HEAD가 같거나 코드 Git diff가 없다는 사실만으로 private seal PASS를 주장하지 않는다. 보완 코드 커밋은 `20e5011cf2cf942e90b7802a51e5ae8096515277`에서 실제 코드 바이트와 대조한다.
4. 동결 환경(Python3.13.5/numpy2.3.5/pandas2.2.3/scipy1.17.0/sklearn1.8.0/joblib1.5.3/threadpoolctl3.6.0), 개발 CSV 및 세 manifest의 SHA와 전체 ID→Fold 배정을 대조한다. C의 기존 OOF·메타데이터·기준선 fingerprint도 원래 design과 결속한다. 홀드아웃과 원본 전체 CSV 내용은 읽지 않는다.
5. 9개 구성의 metadata와 Fold 원장, interruption, 보완 progress/log를 합쳐 attempted/completed/interrupted/unattempted를 따로 센다. 저장 파일 수만으로 시도 수를 결정하지 않는다. `status=running`은 중단 당시 스냅샷일 수 있다.

기존43시도/42완료 패턴이나 보완 원장이 어긋나면 중단한다. 원래 완료 Fold·중단 Fold2·보완 완료 Fold는 재학습하지 않는다. **지금 `run prepare`, `run fit`, `run fit-all`, `complete_unattempted`를 실행하지 않는다.** 확인된 미시도만 새로 수행할 필요가 생기면 누락 증거 회수와 분리해 원장·실행 범위를 먼저 확정한다. 자동 재시도는 없다.

## 3. 원래42개 가용 증거 읽기 전용 감사

원래 design, 중단 패턴, C/T5 비공개 기준선과42개 모델이 모두 확보됐을 때만 기존 감사기를 사용한다. 출력은 존재하지 않는 새 비공개 경로다.

```powershell
python -m experiments.geo_ablation.audit_available --data <DEV_DATA_DIR> --t5 <T5_PRIVATE_DIR> --baseline <BASELINE_PRIVATE_DIR> --output <ORIGINAL_T6_DIR> --destination <NEW_PRIVATE_REAUDIT_DIR>
```

꺾쇠 항목은 회수된 실제 경로로 바꾼다. 이 감사는 분류기를 학습하지 않으며 C 참조 전처리-only fit은 별도로 기록한다. 원래 전체45개를 요구하는 `audit.py`의 조건은 완화하지 않는다.

검사: 원래 seal·source·분할·기준선 SHA, 전체 모델 재로딩/확률, 제거 지역 피처 재유입 없음, C 공통 전처리 행렬 투영과 학습/검증 전 행 일치, 제거 지역값 변경 불변, train-only 중앙값/스케일러/범주, ID/정답/Fold/확률·OOF, 독립 혼동행렬 산술, Fold 평균/SD(ddof=1)/pooled 및 동일 행 오류 전이.

`PARTIAL_PASS_AVAILABLE_EVIDENCE`는 원래42개 가용 증거 범위의 PASS다. 전체 연구 완료로 바꾸지 않는다. 과거 audit.json만 복사하는 것으로 새 독립 감사가 완료되지는 않는다.

## 4. 보완2개 감사 — 실행기 재호출 금지

보완 모델이 있으면 `complete_unattempted.py`를 다시 실행하지 않는다. 이 파일은 감사 앞에2회의 분류기 fit을 포함한다. 그 내부의 **감사 부분만** 새 별도 읽기 전용 검증 경로에서 적용하고 기존 학습·감사 코드는 바꾸지 않는다.

검증 항목:

- `supplement_design.json` SHA가 seal과 일치하고, 기존 design SHA·실제 소스·환경·보완 코드 SHA·원래 전체 증거 fingerprint가 일치하는지 확인한다. pre-fit `classifiers_fitted=0`와 원시 시각·원장도 대조한다.
- progress/로그의 실제2시도/2완료와 Fold 집합{3,4}, 원래43시도 불변, Fold2 재시도 없음, 원래 완료 결과 불변을 확인한다.
- artifact SHA를 검사한 뒤 신뢰한 모델2개를 로드하고 Fold별41,696행, 합계83,392행의 ID/정답/확률/예측을 검증한다.
- 원래 `audit.inspect()`와 독립 `arithmetic()`을 재사용해 공통 C 전처리의 train-only fit2회와 투영·제외 피처·통계·재추론 검사를 수행한다. 모델 파라미터/클래스 순서/수렴·기록 반복수도 대조한다.
- 원래 결과와 보완 결과의 전후 전체 SHA 목록을 비교하고 결과를 새 경로에만 쓴다. tolerance를 완화하거나 결과를 사후 보정하지 않는다. 교차환경 오차가 남으면 별도 미달로 기록한다.

보완 원장이 없으면 보완 시도/완료는 null이다. 모델 일부만 회수돼도 그 범위만 감사하고 누락을 보간하지 않는다.

## 5. 최종 보고 범위와 Git 반영

원래42개와 보완2개가 모두 검증돼도 합계는45시도/44완료/1중단이며 Stratified/R12 전체5Fold 성능은 계속 null이다. 원래8개 완결 구성은 C와 동일5Fold·동일208,480행 기준으로 비교하고, Stratified/R12의4개 완료 Fold는 해당 Fold의 C와만 부분 비교한다. 부분4Fold를 전체 mean/SD/pooled 표에 합치지 않는다.

완결된 비교에만 Macro F1 평균·SD(ddof=1)·pooled F1·Accuracy·등급별 P/R/F1/support·혼동행렬을 기록한다. 실제3등급 Recall, 3→1/3→2와 실제3 support 분모, Fold 일관성, 원본995/비995, 지역26/20/13 및 전체31지역, C 대비 오류 수정·새 오류를 함께 본다. 랜덤 성능 손실과 지역 일반화 이득을 구분하며 과적합/누수의 인과적 입증이나 외부 지진 성능을 주장하지 않는다.

최종 보고서와 집계 JSON만 T6 브랜치에 명시적으로 stage한다. 공개 diff에 개발 CSV/manifest/행별 OOF/joblib/비공개 원시 로그/개인 식별 기록이 없음을 확인한다. 직전 원격 HEAD를 다시 확인해 동시 변경을 보존하고 일반 fast-forward push를 사용한다. 최종 원격 HEAD와 해당 커밋 CI를 확인하되 CI 성공을 실데이터 감사 완료로 승격하지 않는다. main/peter/T1~T5/T4, 기존 코드와 홀드아웃, 후보·배포 모델을 변경하지 않는다.
