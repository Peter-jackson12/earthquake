# T18: LGBM 학습 예산과 leaves의 내부 지역 Fold 선택

상태: 사전 등록. 실데이터 학습과 결과 판정 전이다. T15의 3등급 보정과 결합하지 않는다.

## 질문과 비교

T4/T8/T13의 one-hot LGBM 기준선(300 rounds, num_leaves 31)을 보존한다. T18은 학습 예산 선택 정책과 leaves 선택 정책을 각각 검증한다. 전처리는 기존 C: 학습 행의 age995 중앙값 대치 및 플래그, 수치 스케일링, 지역 및 문자 one-hot CSR float32를 그대로 쓴다. 입력 38개, learning_rate 0.05와 다른 모든 LGBM 설정, library 버전, 예측 argmax는 동일하다.

| 후보 | rounds | leaves | 비교 가능한 정책 |
|---|---:|---:|---|
| B | 300 | 31 | 두 정책 공통 기준 |
| R600 | 600 | 31 | ROUNDS |
| L15 | 300 | 15 | LEAVES |
| L63 | 300 | 63 | LEAVES |

ROUNDS는 B/R600, LEAVES는 B/L15/L63 중에서만 선택한다. 두 요인을 동시에 바꾸거나 외부 점수로 우승 후보/정책을 다시 선택하지 않는다. 두 정책을 대조한 수치는 기술 통계이며 상호작용 효과를 추정하지 않는다.

## 분할과 선택

- 저장된 geo2 manifests seed 42/101/202/303의 외부 5-Fold, 총 20 Fold를 사용한다. 기존 T8/T13 LGBM-O OOF는 설정에 고정된 전체 SHA로 검증하고 읽기 전용으로 재사용한다.
- 외부 학습 행에서만 `(geo_level_1_id, geo_level_2_id)` 기준 StratifiedGroupKFold 3개를 만든다(shuffle=True, seed=42). 작은 후보 수와 합리적인 학습 비용을 위해 T15의 내부 5개 대신 3개를 사전 고정했다. 외부 검증 행/정답은 내부 분할 및 학습/선택에 들어가지 않는다.
- 내부 학습/검증의 각 클래스 support는 30 이상, 지역 그룹은 겹침이 없어야 한다. 실패 시 멈추며 seed를 바꾸지 않는다.
- 모든 후보는 같은 내부 Fold를 공유한다. 후보별 pooled 내부 OOF의 labels=[1,2,3] Macro F1을 계산한다. ROUNDS/LEAVES 별 최대값을 선택하며 정확한 동률은 등록 순서 B→R600 또는 B→L15→L63으로 처리한다.
- 선택을 디스크·SHA 원장·백업에 고정한 뒤 선택된 후보만 외부 학습 데이터 전체로 학습한다. B 선택 시 해당 외부 Fold의 기존 B OOF를 재사용하므로 불필요한 외부 재학습을 하지 않는다. 보정 오프셋은 0, class_weight/ROS/TE/early stopping은 모두 사용하지 않는다.
- 외부 Fold에서 후보별 전수 학습이나 sweep를 하지 않는다. 외부 검증은 선택 정책의 성능 측정에만 쓴다.

기준선 정합성은 T15→T13→T8/T4 설정 검증 사슬, T8 `fit_arm('O')` 전처리/학습 구현의 직접 재사용, 동결 환경 일치 및 봉인된 전체 소스 해시로 확인한다. 후보에서 변경 가능한 키는 `n_estimators`, `num_leaves` 두 개뿐이며 각 비교 팔은 그중 한 개만 다르다. 새로운 외부 B 학습 결과로 기존 OOF를 소급하지 않는다.

## 비용과 실행 통제

내부 학습은 20×3×4=240회. 외부 재학습은 정책별 선택 후보 최대 2개×20=40회로 최대 280회다. 내부 B는 두 정책이 같은 체크포인트를 공유한다. 내부 boosting rounds 합은 90,000회, 외부는 (600+300)×20=18,000회로 합계 108,000회(다중분류 트리 324,000개 상한)다. leaves 63은 회당 비용이 더 들 수 있어 이 상한은 실행시간 추정치가 아니다.

학습 병렬화는 LGBM 내부 2스레드뿐이다. 외부/내부 Fold는 순차 실행한다. 첫 geo2_s42/f0의 최대 14회 학습 시간을 확인하고 전부 실행한다. 부분 실행은 설계 변경이나 새 탐색을 뜻하지 않는다. 실패한 시도도 280회 예산에 포함하며 자동 재시도는 없다. 같은 한도에서 미완료 시도를 명시적으로 재개할 수 있고, 예산이 모자라면 중단한다.

실데이터 prepare/fit은 메인의 단일 실행권 승인 후 실행한다. 테스트는 합성 데이터만 쓴다. 홀드아웃과 팀원 브랜치는 접근하지 않는다. 기존 private 출력과 태그를 덮어쓰지 않는다.

## 봉인과 재개

새 비공개 경로는 `local_reproduction/T18_20261015_lgbm_tuning/run`이다. prepare는 HEAD 전체 SHA/소스 청결성/환경/데이터 계약/manifest/기존 OOF 해시와 내부 분할을 검증하고 모델 0개 상태에서 설계와 백업을 봉인한다. 각 내부·외부 fit는 `attemptN` 새 경로에 모델, 전처리기, 행별 확률 OOF, 지표를 저장한다. 모델 재로딩 확률 허용오차는 1e-12이며 CSV는 %.17g, round_trip으로 정확히 재검증한다. 원본/백업 SHA를 확인한 다음에만 complete로 표시한다.

완료 fit는 재개 시 원본/백업 SHA를 검증한 뒤 건너뛴다. 실패 파일은 보존하며 `--resume-incomplete`로 새 attempt 경로에 기록한다. 실행 중 프로세스가 강제 종료되어 RUNNING.lock이 남으면 프로세스 종료를 확인한 뒤 잠금 파일만 해제한다. 선택 및 평가 기록도 백업 후 원장 기록 사이에 중단된 경우 기존 내용을 검증해 복원한다.

## 사전 판정과 한계

정책별 자기 기준선 B 대비 외부 20 Fold의 Macro F1 평균 차 ≥0.003, 엄격한 Fold 승리 ≥14/20, seed 평균 양수 ≥3/4를 모두 만족하면 해당 선택 정책을 뒷받침한다고 표현한다. 클래스별 Precision/Recall/F1, 혼동행렬과 후보 선택 빈도, fit시간도 기록한다. 이는 제한 후보 집합의 nested 정책 비교이며 최종 모델 교체를 뜻하지 않는다. 같은 개발 행을 4번 재분할한 결과라 20개 독립 표본으로 간주하거나 유의성 주장을 하지 않는다. 다른 미시험 hyperparameter의 우열도 주장하지 않는다.

원본 모델/OOF/manifests는 Git에 올리지 않는다. finalize의 private 집계는 독립 감사 후에만 검토된 reports 파일로 옮긴다. 공유 README와 docs/PLAN은 메인이 관리한다.

## 구현 검증 기록

동결 Python에서 전체 `python -m unittest discover -s tests -v`: 536개 실행, 실패 0, 스킵 33(기존 private 준비 데이터 미보유 통합 검증), 약 68초. T18 신규 합성 검증은 20개다. 외부 정답 변경에 대한 선택 불변, 내부 지역 비중첩과 OOF 완전 피복, 단일요인/기준선 파라미터 정합, 선택된 B의 외부 재학습 생략, 완료 체크포인트 재사용, 중단 후 새 attempt 보존과 백업 단계 복구, 모델·CSV 재로딩, 희소 행렬, pooled 계산과 전체 finalize를 검증했다. 이 기록은 실데이터 prepare/학습/성능 또는 독립 감사 완료를 뜻하지 않는다.

## 실행 명령

worktree `worktrees/t18_lgbm_tuning`에서 동결 Python을 사용한다. `CODE_SHA`는 사전 등록 커밋의 실제 전체 SHA로 대체한다.

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:TEMP='C:/Projects/02_Team_Project/total_earthquake/local_reproduction/T18_20261015_lgbm_tuning/tmp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP
$py='../../local_reproduction/20261014_01/venv/Scripts/python.exe'
$out='../../local_reproduction/T18_20261015_lgbm_tuning/run'
& $py -m experiments.lgbm_nested_tuning.run prepare --data ../../earthquake/data --output $out --code-commit CODE_SHA
& $py -m experiments.lgbm_nested_tuning.run fit-all --data ../../earthquake/data --output $out --schemes geo2_s42 --outer-folds 0
& $py -m experiments.lgbm_nested_tuning.run status --output $out
& $py -m experiments.lgbm_nested_tuning.run fit-all --data ../../earthquake/data --output $out
& $py -m experiments.lgbm_nested_tuning.run finalize --data ../../earthquake/data --output $out
```

정상 부분 실행의 재개는 같은 fit-all 명령이다. 미완료 fit가 있는 중단을 명시적으로 재개할 때만 `--resume-incomplete`를 추가한다. 실제 성능·감사 결과·실행 완료는 추후 별도 보고한다.
