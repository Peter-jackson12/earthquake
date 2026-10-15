# T17 — 전체 개발 Group OOF로 3등급 결정 규칙 b 고정

- 실험 ID: `T17_grade3_rule_freeze_v1`, 브랜치: `peter-exp/t17-rule-freeze`.
- 코드: `experiments/grade3_rule_freeze/`, 합성 테스트: `tests/test_grade3_rule_freeze.py`.
- 전용 비공개 출력: `local_reproduction/T17_20261015_rule_freeze/run/`.
- 현재 상태: 사전등록·구현 단계. 실데이터 실행과 b 선택을 수행하지 않았다.
- 목적은 T15에서 규칙 효과가 확인된 `LGBM_O`, `LGBM_TE_G` 각각의 b를 전체 개발 내부 Group OOF로 한 번 고정하는 것이다. 최종 모델 선정·규칙 채택·최종 모델 fit·홀드아웃 평가는 포함하지 않는다. LR-C 기준선을 유지한다. class_weight는 별도 학습 요인이며 이번 범위에서 제외한다.

## 1. 사전 고정 설계

각 모델별 `pred = argmax(p · (1,1,exp(b)))`, `b ∈ {0,0.1,…,2.0}`를 유지한다. 정확 동률이면 가장 작은 b를 선택한다. argmax 동률은 원래 클래스 순서 `[1,2,3]`를 따른다.

개발 데이터 208,480행을 geo2 합성 그룹 `(geo_level_1_id, geo_level_2_id)`으로 나눈 5-Fold OOF를 사용한다. seed 42/101/202/303 각각에서 모든 개발 행을 정확히 한 번 검증 예측했으므로, 각 seed의 OOF는 전체 개발 데이터를 내부 Group CV로 나눈 확률 자료다. T15에서 사용한 바깥 OOF를 이 단계에서는 **전체 개발 데이터의 내부 OOF**로 재사용한다. T15의 바깥 학습 행만 포함한 내부 OOF를 전체 개발 OOF로 오인하지 않는다.

목적함수는 각 seed에서 모든 행을 풀링한 labels=[1,2,3] Macro F1을 계산한 뒤, **네 seed 점수의 산술평균**을 최대화하는 것이다. Fold 평균도, 네 OOF 행을 합쳐 계산한 단일 Macro F1도, 확률 평균 후 Macro F1도 아니다. 모델당 b는 한 개이며 seed별 b를 별도로 채택하지 않는다. Fold별 곡선은 진단용으로만 남긴다. 그리드 상한 도달 여부를 보고하되, 상한 도달 시 확장·재탐색하지 않는다.

## 2. 재사용 조건과 경계

재사용 파일은 T15 `config.json`의 geo2 네 scheme에서 두 팔에 해당하는 확률 OOF 8개와 manifest 4개다. seed42의 LGBM-O는 T8, LGBM_TE_G는 T11이며 나머지 seed는 T13이다. 전체 경로·64자 SHA는 T17 `config.json`에 복사해 봉인하고, 실행기가 T15 설정과의 일치를 다시 검사한다. T15의 조정된 예측은 입력으로 쓰지 않는다.

`prepare`와 재개 검증에서 개발 데이터 계약(개발 CSV·split_manifest SHA), 동결 환경, manifest SHA, OOF SHA, 정확한 ID 집합·중복 없음·라벨 일치·Fold 일치·유한한 비음수 확률·행합 1·`argmax(p)==y_pred`를 확인한다. manifest는 각 그룹의 Fold 유일성, 학습/검증 ID·그룹 교집합 0, 5개 Fold 전체 커버리지와 기존 클래스 지지 게이트를 검사한다. 하나라도 실패하면 중단한다. 재학습 대체·seed 대체·확률 수정은 하지 않는다.

**새 모델 학습은 0회**다. 모델 파일을 로딩하거나 전체 개발 최종 모델을 fit하지 않는다. 기존 OOF의 출처와 기존 독립 감사는 T8·T11·T13·T15 보고서에 의존한다. 이 단계의 SHA 검증은 과거 모델 학습 과정을 새로 증명하는 절차가 아니다. 새 자료를 만들지 않고 검증된 OOF를 재사용하는 제한이다.

## 3. 선택 통계와 독립 성능의 구분

b를 선택한 동일 OOF에서 최대 점수·보정 전 대비 차이를 보고해도 **선택용 개발 통계**다. 독립 성능 추정, 개선 재확인, 팔 간 우열·최종 후보 판정, T15의 독립적인 반복 검증으로 해석하지 않는다. b 선택에 사용한 라벨로 다시 평가하는 낙관성이 있다.

분할과 팔은 T15 결과를 보고 선정됐고, 같은 OOF도 이전 연구에서 사용됐다. 따라서 사전등록은 이번 b 고정 절차를 사전에 묶는 것이며, 연구 전체를 비적응적인 새 검증으로 만들지 않는다. 네 seed는 같은 개발 행을 반복 예측한 것이므로 독립 표본 네 개가 아니다. 지역 일반화 분포를 기준으로 정한 b이므로 이미 본 지역·새 학습 설정·튜닝된 별도 모델에 자동으로 이전하지 않는다. 전체 개발 fit 모델과 OOF 모델의 학습 크기 차이도 남는다.

## 4. 봉인·체크포인트·재개

1. 전용 계획·설정·코드·합성 테스트를 커밋하고 메인 검토 후 push한다.
2. 컨트롤타워 승인 후 `prepare --code-commit <40자 HEAD>`로 소스 SHA·환경·데이터 계약·입력 검증 결과를 `design.json`에 봉인한다. 기존 출력 디렉터리가 있으면 거부한다. 설계·봉인을 `backup/`에 복사한다.
3. `freeze-all`은 한 프로세스만 실행한다(`RUNNING.lock` 배타 생성). 각 seed/팔의 풀링 곡선·Fold 곡선과 그리드별 클래스 P/R/F1·혼동행렬을 집계 체크포인트로 저장하고 JSON 재로딩·SHA·백업을 검증한 뒤 원장에 완료를 기록한다. 이어 두 모델별 고정 b를 같은 절차로 저장한다. 완료 단위 10개(8개 곡선, 2개 고정 b)는 원본·백업 해시만 확인하고 재계산하지 않는다.
4. 중단 후 동일 `freeze-all`로 재개한다. 미완료 파일은 `.incomplete_*`로 보존한다. 잠금이 남았다면 해당 프로세스가 종료된 사실을 메인이 확인한 뒤에만 잠금을 제거한다. 자동 재시도와 기존 완료 파일 덮어쓰기는 없다.
5. `finalize`는 모든 10단위의 해시와 두 b의 재계산 일치를 확인한 뒤 비공개 `analysis.json`과 백업을 만든다. 기존 집계 파일은 덮어쓰지 않는다.
6. 실행기를 import하지 않는 독립 감사에서 원본 입력 해시·OOF/그룹 조건·전체 21점 목적함수·동률 처리·선택 b·원장·백업을 처음부터 검증한다. 감사 후에만 집계 보고서와 JSON을 `reports/`에 커밋한다. 이 계획·코드는 결과 보고서가 아니다.

학습이 없어 Fold별 신규 모델 체크포인트는 적용되지 않는다. 대신 기존 5-Fold OOF의 출처/그룹 조건을 검증하고 seed/팔 집계를 체크포인트로 삼는다. 원시 데이터·OOF·manifest를 새 출력에 복제하지 않는다. 예상 비용은 새 fit 0회, 확률 CSV 재로딩·그리드 점수 계산 수 분(미측정), 새 집계 디스크 10MB 미만(추정)이다.

## 5. 승인 후 실행 명령

작업 디렉터리는 이 worktree 루트다. Python은 동결 환경의 `../../local_reproduction/20261014_01/venv/Scripts/python.exe`를 쓴다. 설치/업그레이드는 하지 않는다.

```powershell
$py = '../../local_reproduction/20261014_01/venv/Scripts/python.exe'
$out = '../../local_reproduction/T17_20261015_rule_freeze/run'
$sha = git rev-parse HEAD
& $py -m experiments.grade3_rule_freeze.run prepare --data ../../earthquake/data --output $out --code-commit $sha
& $py -m experiments.grade3_rule_freeze.run freeze-all --data ../../earthquake/data --output $out
& $py -m experiments.grade3_rule_freeze.run finalize --data ../../earthquake/data --output $out
```

합성 테스트와 저장소 필수 검증은 `& $py -m unittest discover -s tests -v`다. 홀드아웃 경로는 어떤 실행에서도 열지 않는다. `preprocess.py`, 공유 README/PLAN, 팀원 브랜치, 기존 출력·태그를 바꾸지 않는다.
