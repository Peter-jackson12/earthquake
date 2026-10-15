# T19 고정 50:50 확률 앙상블 사전 계획

컨트롤 타워 승인으로 가중치·기술 기준을 동결한 뒤, 공식 T19 분석 전에 이 계획과 코드·합성 검사를 커밋한다. 개인 연구 위치는 `worktrees/peter_rules`의 `peter`다. 새 학습, 확률 재추론, 가중치 탐색, b 보정, T17 규칙 및 T18 튜닝 결합은 하지 않는다.

절차 예외: 독립 감사 준비 단계의 입력 검사에서 기준 동결 후·코드 커밋 전에 원본 8개 OOF를 이용한 혼합 지표 산술이 1회 선행됐고, 비공개 `reference_inputs_receipt.json`을 생성했다. 그 뒤 기준·가중치·설계 선택은 변경하지 않았다. 원본 영수증을 보존하며 이 작업을 완전한 사전등록으로 주장하지 않는다. 공식 실행기의 prepare/analyze와 선행 감사 산술을 구분하고 최종 보고에도 이 예외를 명시한다.

## 질문과 고정 입력

동일 개발 행과 geo2 Group 5-Fold에서 LGBM_O와 LGBM_TE_G의 오류가 얼마나 겹치고, 두 원본 확률의 고정 산술 평균이 각 모델보다 나은지 살핀다. seed 42는 T8/T11, seed 101·202·303은 T13의 저장 검증 OOF만 사용한다. 입력 경로·OOF 8개·manifest 4개 SHA와 환경은 `experiments/fixed_ensemble/config.json`에 고정한다.

개발 CSV와 원래 분할 manifest는 `configs/data_contract.json`의 SHA와 행수 계약을 검사한다. 원시 입력·홀드아웃·모델 바이너리는 읽지 않는다. 홀드아웃 디렉터리 접근을 금지한다. 원본 design/seal/ledger와 각 완료 fold JSON·검증 예측의 SHA를 봉인하고, OOF가 완료 fold 검증 확률과 정확히 같은지 검사한다. 저장 모델 재추론은 하지 않으므로 과거 모델 재추론 감사와 이번 OOF 산술 감사를 구분한다.

모든 OOF는 `building_id`로 정렬한 뒤 개발 전체 정확한 1회 커버리지, 중복·누락 없음, 정답·fold 일치, 각 geo2 그룹의 단일 fold 배정, fold마다 각 클래스 지지 30 이상을 요구한다. 확률 열 순서는 `p_grade_1`, `p_grade_2`, `p_grade_3`이며 유한·[0,1]·행합 오차 1e-10 이내와 원본 argmax 라벨 일치를 요구한다. CSV는 `float_precision='round_trip'`로 읽는다. 실패 시 기존 근거를 보존하고 중단하며 입력·허용오차를 사후 변경하지 않는다.

## 고정 규칙과 측정

`p_ensemble = 0.5 * p_O + 0.5 * p_TE_G`. labels=[1,2,3], argmax 완전 동률은 첫 열(낮은 등급)을 택한다. b=0의 원본 확률이며 어떤 탐색도 하지 않는다. 신규 fit은 0이다.

4 seed×5fold=20 matched fold에서 각각 두 baseline과 앙상블의 Macro F1, Accuracy, 클래스별 Precision/Recall/F1/support, 3×3 혼동행렬을 저장한다. 각 seed 전체 OOF pooled 지표는 fold 평균과 분리한다. 비교 Δ는 항상 앙상블−같은 fold/seed의 baseline이다. 20fold Δ 평균·표본 표준편차(ddof=1)·양수/동률/음수 fold 수와 4개 seed pooled Δ를 남긴다. 같은 개발 행을 seed별로 네 번 반복해 독립 833,920개 표본으로 합산하지 않는다.

3→1·3→2의 분모는 실제3등급 수, 2→3의 분모는 실제2등급 수다. 오류 수·분모·비율을 모두 기록한다. 두 모델의 둘 다 정답·둘 다 오류(오류 교집합)·O만 정답(TE 구제)·TE만 정답(O 구제)의 서로 배타적인 네 범주와 각 범주의 앙상블 정답/오류 수, 모델 예측 불일치, 오류 합집합을 fold/seed pooled 각각 남긴다.

위 정답/오류 네 범주는 실제1·2·3등급별로도 분리하고 각 분모와 비율을 남긴다. 3→1·3→2·2→3 사건 각각에 대해서는 해당 실제등급 안에서 두 모델 모두 해당 오류·O만 해당 오류·TE만 해당 오류·둘 다 해당 오류 아님의 네 범주를 따로 저장한다. 각 범주의 전체 개발행 대비 비중과 실제등급 대비 비율, 앙상블의 해당 오류/실제 정답/다른 오류 수도 구분한다. 각 baseline 대비 사건 제거·새 사건 유입·순감소를 남기며, 사건 제거는 실제 정답 전환과 다른 오류 전환으로 나눠 3→1에서 3→2로 바뀐 경우를 완전 구제로 오해하지 않게 한다.

## 사전 기술 기준

각 baseline별로 다음 네 항목을 모두 만족해야 해당 baseline 대비 기술 기준 충족으로 표시한다.

1. 20 matched fold 평균 ΔMacro F1 ≥ **0.003**.
2. ΔMacro F1이 엄격히 양수인 fold ≥ **15/20**.
3. 모든 4 seed의 pooled ΔMacro F1이 엄격히 **>0**.
4. 모든 4 seed의 pooled 실제3등급 Recall Δ ≥ **−0.01**.

두 baseline 각각의 판정과 둘 모두 충족 여부를 구분한다. 이 기준은 사전 선택된 개발자료에서의 기술적 기준이며 통계적 유의성·일반화 보장·독립 성능·최종 모델 선정·운영 채택의 근거가 아니다. 기준 실패 후 가중치를 바꾸거나 재튜닝하지 않는다. 기준은 분석 결과 확인 전 컨트롤 타워가 승인했다.

## 실행·체크포인트·감사

동결 Python `../../local_reproduction/20261014_01/venv/Scripts/python.exe`를 사용하고 설치·업그레이드를 금지한다. TEMP/TMP는 T19 비공개 temp 경로, 입출력은 UTF-8이다. 실행기는 학습 실행기를 import하지 않는다. `preprocess.load_development`와 수치·지표 라이브러리만 사용한다.

먼저 계획·코드·합성 검사를 커밋한 전체 SHA로 `prepare`한다. `SOURCE` 파일 목록과 SHA, 코드 SHA, 환경, 데이터 계약, 원본 provenance, 입력 검증을 `design.json`과 `design_seal.json`에 봉인하고 백업한다. 실행 경로는 `../../local_reproduction/T19_20261015_fixed_ensemble/run`만 허용한다. 실행 순서는 `prepare --code-commit <full SHA>` → `analyze` → `finalize`이며 각 명령에 `--output`을 지정한다.

각 seed는 작은 집계 JSON과 SHA 영수증을 완료 즉시 저장·재로딩·별도 백업한다. 원장은 불변 `ledger.json`과 append-only 단위 영수증이며, 완료 단위는 재계산하지 않는다. 미완료 orphan은 고유 이름으로 보존하고 재개한다. 기존 run·최종 집계·백업은 덮어쓰지 않으며 실행 잠금을 사용한다. 봉인·원본 SHA 변경이면 중단한다.

독립 감사자는 실행기를 import하거나 모델을 로드하지 않고 입력 SHA·정렬·고정 평균·argmax·각 지표·오류 분모·판정을 별도 산식으로 확인한다. 감사 PASS 전 `reports/`에 결과를 게시하지 않는다. PASS 후 집계 보고와 JSON만 공개 후보로 준비하며 원시 데이터·행별 OOF·manifest·모델·비공개 전체 경로는 공개 Git에 올리지 않는다. 공유 문서·커밋·푸시는 컨트롤 타워가 담당한다.
