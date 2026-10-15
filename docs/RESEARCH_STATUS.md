# Earthquake 개인 연구 현황 (Research Status)

- 기준 확인일: **2026-10-15 (KST)**, 저장소 `Peter-jackson12/earthquake`. 개인 연구는 `worktrees/peter_rules`의 `peter`에서만 진행한다. 새 실험 브랜치/worktree는 만들지 않는다.
- T0~T18 코드와 기존 보고서는 `peter`에 통합했다. 통합 확인 커밋 `050130bd9c08f066e61d360e4298f0d200c7167d`: 전체 테스트 564개 실행·실패 0·스킵 33, GitHub Synthetic/Dashboard CI 성공. **코드 통합과 실험 완료는 별개다. 현재 HEAD는 매번 다시 조회한다.**
- 목적: 세션·모델 간 연구 인계의 시작점. **결정 근거는 연결된 실제 코드·보고서·OOF/검증 산출물**이며 이 문서만으로 검증을 생략하지 않는다.
- 팀 공용 `main`과 개인 `peter`를 구분한다. 팀원 `EY`·`jjjjin`·`sy`는 변경하지 않는다. 이 문서는 팀 최종 선택이나 배포 승인 문서가 아니다.

## 현재 실행 상태 (2026-10-15 12:15 KST 관측)

| 연구 | 확인 상태 | 근거·다음 단계 |
|---|---|---|
| T5~T15 | 기존 보고서·감사 증거 보존 | [README 실험 목록](../README.md)의 보고서 참조. 아래 T1~T4 상세는 당시 결과의 기록이다. |
| T16 RF 요인 분리 | 원장상 8/80 완료·1 Fold running, 실행 관리자가 학습 관리 중 | [사전 등록](T16_RF_FACTORS_PLAN.md). 전체 학습 종료 → 집계 → 독립 감사 → 보고. 완료 수는 관측 시점 값이며 미완료 Fold로 결론을 내리지 않는다. |
| T17 결정 규칙 고정 | 완료·독립 감사 PASS, 신규 fit 0 | [보고](../reports/GRADE3_RULE_FREEZE_20261015.md). LGBM-O b=0.5, LGBM_TE_G b=0.3. 같은 개발 OOF의 선택용 통계이며 독립 성능이 아니다. |
| T18 LGBM 내부 튜닝 | 봉인 완료·학습 대기 | [사전 등록](T18_LGBM_TUNING_PLAN.md). T16 종료 후 단일 실행자로 내부 240 fit·외부 최대 40 fit. |

LR-C 연구 기준과 기존 최종 모델은 유지한다. 홀드아웃 평가, 최종 모델 선정, T17 규칙과 T18 튜닝의 결합은 승인된 연구 범위에 포함되지 않는다. 실행 중 T16이 읽는 임시 폴더는 학습·감사 종료까지 보존한다.

## 문서 역할과 주장별 근거

README는 결과 입구, AGENTS는 현재 규칙, 이 문서는 상태·근거·후속 절차, PLAN은 남은 작업과 완료 이력, 각 실험 계획은 사전등록 당시 설계, reports는 감사된 집계 결과다. **T16/T17/T18 계획의 옛 브랜치·최초 prepare 명령·미실행 표기는 당시 기록이다. 봉인 파일은 그대로 보존하며 현재 실행 지시로 사용하지 않는다.**

아래 관측일은 2026-10-15다. 이번 문서 점검은 작은 문서·코드·원장·기존 감사 영수증 대조이며 원시 자료나 저장 모델 재감사가 아니다. 로컬 경로는 작업 공간 루트 기준이다. `기존 확인`은 이번 재실행을 뜻하지 않는다.

| 핵심 주장 | 대상·근거 | 현재 판정 | 닫기 조건·담당 |
|---|---|---|---|
| 개인 작업 위치가 하나다 | `peter`, `AGENTS.md`, Git worktree 목록 | 현재 확인. T16 실행용 detached 폴더만 예외 보존 | 컨트롤 타워: 종료·감사 후 임시 폴더 정리 |
| 팀 문제 정의가 개인 연구에 반영됐다 | `cc5db91` → `de4b9dc`, README/PLAN | main·peter 모두 조상 관계 확인, 해당 docs 브랜치 삭제 | 컨트롤 타워: 내용 변경 때 재대조 |
| 통합 코드 테스트가 통과했다 | `050130b`, 로컬 `local_reproduction/session_20261015_control/peter_integrated_tests.log`, [Synthetic CI](https://github.com/Peter-jackson12/earthquake/actions/runs/38674116222), [Dashboard CI](https://github.com/Peter-jackson12/earthquake/actions/runs/38674116337) | 기존 564개·실패 0·스킵 33. 문서 변경 후 HEAD의 실자료 검증은 아님 | 코드 변경 담당: 영향 테스트와 해당 revision CI |
| T16 설계가 봉인됐다 | `cb8c14e`, `local_reproduction/T16_20261015_rf_factors/run/design.json`, [실행기](../experiments/rf_factors/run.py) | 봉인 기록 확인, 성능 결론 대기 | 실행 관리자: 봉인 일치·80 Fold 완료 후 독립 감사 |
| T16 실행이 완료됐다 | 같은 폴더 `ledger.json`, `RUNNING.lock` | **미완료**: 관측 당시 완료 8·running 1 | 실행 관리자: 80/80, 프로세스 종료·잠금 해제, finalize |
| T17 규칙이 고정됐다 | `3dc5daf`, [보고](../reports/GRADE3_RULE_FREEZE_20261015.md), 로컬 `local_reproduction/session_20261015_control/t17_audit_result.json` | 기존 독립 감사 PASS 재확인, 신규 fit 0 | 완료. 후속 변경 시 새 설계·증거 필요 |
| T17 점수가 독립 성능이다 | 같은 보고의 선택 곡선과 한계 | **성립하지 않음**: 같은 OOF로 b를 선택한 통계 | 최종 선정·독립 평가를 별도 설계하기 전 주장 금지 |
| T18 튜닝이 완료됐다 | `310c497`, `local_reproduction/T18_20261015_lgbm_tuning/run/{design,ledger}.json`, [실행기](../experiments/lgbm_nested_tuning/run.py) | **봉인 완료·fit 0**, 성능 미검증 | 실행 관리자: 내부 240 fit·외부 최대 40 fit, 20단위 선택·평가, 독립 감사 |
| 학습과 검증 정보가 분리된다 | T16/T18 계획 및 실행기, [데이터 계약](../configs/data_contract.json) | 설계·구현 근거 있음. 진행 중 실험의 전체 산출물 감사 대기 | 독립 감사 담당: 그룹·ID·내부 선택·예측·모델 재추론 확인 |
| 재개가 결과를 보존한다 | 실행기 원장·백업·재로딩 로직 및 합성 검사 | 구현 근거 있음. 이번에 강제 중단·복구 실험은 하지 않음 | 장애 시 실행 관리자: 원본 보존·기존 완료 Fold 재학습 없음 검증 |
| 전체 파이프라인이 검증됐다 | 학습 → 집계 → 독립 감사 → 보고 | **아직 아님**: T16/T18 학습·감사 대기 | 각 감사 PASS와 revision·설계 결속된 집계 보고 필요 |
| 홀드아웃·최종 모델을 바꾸지 않는다 | AGENTS, T0 보고, 후속 계획 | 현재 작업 범위로 유지. 이번 파일 변경은 문서에 한정 | 컨트롤 타워: 새 평가 설계·사용자 결정 전 접근/선정 금지 |

## 현재 후속 절차 — 문서상 점검, 명령 실행 없음

1. 컨트롤 타워가 `peter` HEAD·변경 파일·담당자를 확인한다. 실행 관리자는 하나이며 새 학습을 별도로 시작하지 않는다. 코드 통합과 실제 학습 완료를 구분한다.
2. 작업 위치는 `worktrees/peter_rules`. 동결 Python은 작업 공간 루트의 `local_reproduction/20261014_01/venv/Scripts/python.exe`다. `TEMP/TMP`는 별도 workspace 폴더, `PYTHONIOENCODING=utf-8`을 사용한다. 환경 설치·업그레이드와 이미 봉인한 출력의 `prepare` 재실행을 하지 않는다.
3. 작은 `ledger.json`·로그 끝부분·`RUNNING.lock`과 원 실행 관리자 정보를 함께 확인한다. `status`의 완료 수, 샌드박스에서 PID 미조회, 타 세션의 `Unknown process id`만으로 종료를 판정하지 않는다. 잠금 파일을 임의 삭제하지 않는다.
4. T16은 완료 80/80과 실행 프로세스 종료·잠금 해제를 확인한 뒤 `-m experiments.rf_factors.run finalize --data ../../earthquake/data --output ../../local_reproduction/T16_20261015_rf_factors/run`을 실행한다. 이어 `../../local_reproduction/T16_20261015_rf_factors/audit/audit.py --destination <존재하지 않는 결과 JSON 경로>`로 독립 감사한다. PASS 전에는 성능 보고를 확정하지 않는다.
5. T16 감사 PASS 후 T18을 시작한다. 최초 비용 확인은 `-m experiments.lgbm_nested_tuning.run fit-all --data ../../earthquake/data --output ../../local_reproduction/T18_20261015_lgbm_tuning/run --schemes geo2_s42 --outer-folds 0`. 확인 후 같은 명령에서 마지막 두 필터를 제거하여 나머지를 진행한다. 완료 Fold는 재사용한다.
6. T18은 내부 240 fit·외부 최대 40 fit 및 20단위 선택·평가, 실행 종료·잠금 해제를 확인한 뒤 동일 모듈의 `finalize`를 실행한다. 감사는 `../../local_reproduction/session_20261015_control/audit_t18.py`다. 감사 스크립트의 현재 입출력과 덮어쓰기 방지를 실행 전에 확인한다. 현재 기본 출력은 `t18_audit_result.json`이며 기존 파일이 있으면 덮어쓰지 않고 중단한다.
7. 봉인 불일치, 입력 계약 실패, 중복 실행, 예산 초과, 감사 실패가 있으면 다음 단계 진입을 중단한다. 원본·실패 로그·기존 출력은 보존하고 원인과 재개 가능 범위를 확인한다. 봉인 수정이나 재학습으로 실패를 숨기지 않는다.
8. 독립 감사 PASS 후에만 집계 보고와 JSON을 `peter`에 반영한다. 코드/설계 revision, 비교 분할, Fold 평균과 pooled OOF, 실제 3등급 오류 분모 및 한계를 함께 기록한다. 컨트롤 타워가 검토·커밋·푸시하고 이 문서의 상태·관측시각·근거를 갱신한다.

이 절차는 문서와 CLI 정의를 대조한 결과다. 이번 점검에서 학습·finalize·실자료 재생·장애 복구를 실행하지 않았다. 실제 T16/T18 종료와 감사가 남아 있다.

문서 수정 후 로컬 링크 87개와 T16 31개·T18 37개 봉인 SOURCE의 SHA 일치를 확인했다(2026-10-15 12:17 KST). 검사 영수증은 로컬 `local_reproduction/session_20261015_control/doc_alignment_check_result.json`이다. 전체 학습이나 실자료 파이프라인을 다시 실행한 결과가 아니다.

## 고정 데이터·평가 계약

- 데이터: 2015 네팔 지진 건물 260,601행, 입력 후보 38개, 정답 `damage_grade`(1/2/3), 추적용 `building_id`는 입력 제외.
- 개발 208,480행; 과거 홀드아웃 52,121행. 개발 **Stratified 5-Fold, seed=42**, Fold 학습 166,784/검증 41,696행; 주지표 **Macro F1**.
- 원본 SHA-256: `87abd2bc2b05d3e687bd176959967fd38334fff607e6afeb2d20d0d899c13e97`
- 개발 SHA-256: `8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1`
- 분할 manifest SHA-256: `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721`
- 계약 파일: [configs/data_contract.json](../configs/data_contract.json), [preprocess.py](../preprocess.py).
- **과거 홀드아웃은 개인 최종 평가에 이미 사용했다.** 후속 연구에서 재평가·선택·튜닝에 사용하지 않으며 새 독립 최종 성능으로 주장하지 않는다.
- 공개 Git에는 원본·개발 CSV, 라벨 포함 행별 OOF, 비공개 모델/바이너리를 넣지 않는다. 실제 접근·파일 SHA를 확인하지 못하면 검증/학습을 완료했다고 표시하지 않는다.

## 과거 연구 기록 — T0~T4 (2026-10-14)

| 트랙 | 연구 내용 | 2026-10-14 확인 상태 | 기준 근거 |
|---|---|---|---|
| T0 기존 개인 모델 | v1.0.1 + 무가중치 LR C=1, 개인 최종 평가/대시보드 | **과거 완료**, 이번 새 개발 연구의 독립 홀드아웃은 아님 | [개인 선택](PERSONAL_MODEL_SELECTION_20261013.md), [개인 최종 보고](../reports/PERSONAL_FINAL_REPORT_20261013.md) |
| T1 `age=995` 전처리 | LR 고정 A/B/C, D/E1/E2/E3/F1/F2 | **9개 설정 CV 비교·감사 완료** | [A/B/C](../reports/AGE995_EXPERIMENT_20261013.md), [D/E/F](../reports/AGE995_FOLLOWUP_20261013.md) |
| T2 오류 진단 | C와 E2 예측 비교, 비995·지역·구조·연령 오류 | **OOF 사후 진단·감사 완료** | [오류 진단](../reports/AGE995_DIAGNOSTICS_20261014.md), [연구 후보](../experiments/age995/RESEARCH_CANDIDATE.json) |
| T3 특성 조합 실험 | C 기반 G1~G5(연령 구간, 지역×구조, 지역×연령, 구조×연령) | **별도 세션 결과 추가 커밋 `a4fc5c8` 확인**. T4 세션에서는 G1~G5 OOF를 재감사하거나 결과를 비교 후보로 사용하지 않음. | [G 계획](../experiments/age995/interactions/PLAN.md), [설정](../experiments/age995/interactions/config.json), [코드](../experiments/age995/interactions/run.py) |
| T4 모델 계열 비교 | LR 대비 XGBoost/LightGBM, A와 C 비교 | **기존 환경 20 fit·독립 OOF 감사·모델 재추론 완료**. 교차환경 T4 재학습 CV 4/4 재현, 기존 XGB 저장모델 5/20개 확률 엄격 기준 초과(분류 변경 0). 기존 LR A/C 재사용. | [완료 보고](../reports/MODEL_FAMILY_COMPARISON_20261014.md), [실행·검증 요약](../reports/model_families_v1/execution_summary.json), [교차환경 보충 보고](../reports/MODEL_FAMILY_REPRODUCTION_REVIEW_20261014.md) |

위 T0~T4 표는 2026-10-14 당시 확인 기록이다. 현재 연구의 진입점은 상단 실행 상태와 README 실험 목록이다. 다음 세션은 최신 HEAD와 실제 원장·감사를 확인하며, 동시 세션의 변경을 덮어쓰지 않는다.

## T1/T2 검증된 결론 (LR 조건부)

- LR `A` (995 원본 유지): **0.683677**; `B` (원본+표시): **0.687129**.
- LR `C` (각 학습 Fold의 비995 연령 중앙값 **15**로 995만 대체 + `age_is_995`): **CV Macro F1 0.688687**, 표준편차 0.001508.
- LR `E2` (학습 Fold 비995 연령 q99 **90**으로 995만 상한 처리 + 같은 표시): **0.688664**, 표준편차 0.001265. C보다 높은 Fold도 2/5개. 통계적 우월·동등성을 입증한 것은 아님.
- D/F1/F2의 학습 행 삭제는 995 검증 건물 **1,125개 전부를 3등급으로 예측**하는 부작용. 995가 공식 결측/오류 코드인지는 **미확정**.
- C와 E2의 전체 OOF 예측 일치율 **99.6522%**. **C는 후속 LR 연구 기준이며 다른 모델 계열의 최적 전처리나 기존 최종/배포 모델로 확정된 것이 아님.**
- C의 3→1 오류 452건 중 **447건**은 비995. 지역 코드 26/20/13에 **292건(64.60%)** 집중. 사후 개발 OOF 패턴이며 독립 재현이나 원인 확정이 아님.
- 당시 T3는 **기존 LR + C의 특성 조합**, T4는 별도 모델 계열을 탐색했다. T3 완료 결과는 [보고서](../reports/AGE995_INTERACTIONS_20261014.md)에 있으며 서로 다른 요인으로 해석한다.

## T4 검증된 결과와 인계

- 신규 XGB A/C: CV Macro F1 **0.637946 / 0.638558**; LGBM A/C: **0.669830 / 0.669847**. LR A/C **0.683677 / 0.688687**는 기존 OOF 검증 후 재사용했다.
- 4설정×5Fold=20회 실학습 완료. 6설정 OOF **1,250,880행**, 저장20모델의 검증 예측 **833,920행** 재추론 일치. 별도 산식 감사 및 sklearn 집계 재검증 PASS.
- 전체 3→1은 줄었지만 3→2가 늘고 3등급 Recall은 낮아졌다. XGB-C는 지역13/20에서 실제3등급 Recall **0%**. 새 모델을 안전성 개선 또는 최종 모델로 채택하지 않는다.
- XGB C−A의 CV 평균은 +0.000612지만 995 집단 F1은 하락했다. LGBM C−A 평균은 +0.000017에 불과하며 995 Recall3은 하락했다. **LR에서 선택한 C를 트리에 자동 채택하지 않는다.**
- 원핫/수치 표준화 CSR 표현, 300 rounds 고정 설정에 한정한 결과다. 라이브러리 전체의 우열을 확정하지 않는다. T4 당시 native categorical 등은 미실행이었다. 후속 범주형 비교는 T8/T9로 완료했으며 [README 실험 목록](../README.md)을 따른다.
- 당시 T4 전용 브랜치에서 작성한 기록이다. 현재는 `peter`에 통합했으며 원래 결과와 출처 커밋을 보존한다.

## 로컬 교차환경 재현 감사 보충 (2026-10-14)

- 출처: 사용자 제공 `REPRODUCTION_REPORT.md`(2026-10-14 KST)와 이 브랜치의 기존 T4 완료 보고·감사 코드·`verification.json` 대조. **이 기록 작업에서 로컬 비공개 원시 로그·모델·OOF ZIP 전체를 새로 독립 실행/감사하지는 않음.**
- 보고된 로컬 재현: 전체 **38개 실행·190 Fold** 수행(별도 BLAS 진단 1회 제외). 사전 고정 엄격 F1 기준 `1e-6`에 **16개 실행 일치**, **LR 계열 22개는 `DIFFERENCE_UNDER_REVIEW`**. 기존 age995 OOF **1,876,320개·45 Fold 산술 감사 PASS**.
- T4 신규 재학습 XGB A/C·LGBM A/C **4설정 모두 원래 CV Macro F1 수치 재현 통과**. 기존 T4 원시 OOF 및 집계 산술 검증 PASS.
- 기존 T4 저장 모델의 Windows 교차환경 재추론: 전체20개 중 **XGBoost 5개에서 사전 확정 절대확률 오차 `1e-7` 기준 초과**, 최대 **`1.1920928955078125e-7`**. 전체 모델의 **예측 등급 변화 0건**. 나머지15개는 해당 기준 통과. **기존 환경 내부 모델 재추론 PASS(`최대 1.11e-16`)와 이 교차환경 미달을 명확히 구분**한다.
- LR 차이의 원인으로 OS·BLAS·wheel 수치 환경 차이가 의심된다. 동일 Fold의 CSR 입력이 일치해도 `OPENBLAS_CORETYPE=Haswell` 대 `Nehalem` 진단에서 최적화 반복수·예측이 달라졌으나, 과거 Linux 실행 환경의 정확한 빌드 및 모델 계수가 없어 역사적 차이를 단일 원인으로 확정할 수 없다.
- **기존 5-Fold 점수·OOF·감사 영수증과 LR-C 연구 기준·후보 결정은 그대로 보존**한다. 사후 확률 허용오차 완화, 재학습/튜닝, 홀드아웃 접근·평가, 모델 재선정 금지.
- 상세 결과·검증 한계·최소 후속 확인 범위는 [T4 로컬 재현 감사 보충 보고서](../reports/MODEL_FAMILY_REPRODUCTION_REVIEW_20261014.md)를 참조. 미확보 과거 G/age995 모델 바이너리는 원래 재추론 완료로 표시하지 않는다.

## 병렬 작업 및 인계 규칙

1. 세션 시작 때 **peter HEAD·최근 변경 파일·현재 실행 관리자와 원장**부터 다시 확인한다. 이어지는 보고서의 출처 커밋과 증거가 맞는지 확인한다.
2. 컨트롤 타워는 공통 문서·통합을, 실행 관리자는 T16→감사→T18→감사와 전용 보고서를 담당한다. 같은 파일을 동시에 수정하지 않는다. 실행 중 봉인된 코드·설정·계획을 변경하지 않는다.
3. **모델 구조 효과(T4)**와 **특성 상호작용 효과(T3)**, **995 처리 효과(T1)**를 분리한다. 한 번에 둘 이상 변경한 성능 차이를 한 원인에 귀속하지 않는다.
4. 기존 실험은 기준 데이터·Fold·모델 설정·OOF SHA가 일치할 때만 재사용한다. 과거 점수를 신규 실행으로 표시하지 않는다.
5. 모델 선택은 개발 데이터 CV에서만 수행한다. 전처리 fit, 범주 인코딩, 샘플링, 학습/검증 선택에 쓰는 통계는 학습 Fold 안으로 제한한다.
6. 평가에 **CV Macro F1 평균/표준편차, pooled OOF 구분, 등급별 P/R/F1·지지 수, 혼동행렬, 3→1/3→2, 실제 3등급 Recall, 995/비995 그룹**을 함께 기록한다. Micro F1=Accuracy는 보조지표.
7. 결과가 없다면 **미실행**, 코드·테스트만 있으면 **구현/합성 검증**, 실제 학습과 OOF 재감사가 끝났을 때만 **성능 검증 완료**로 표시한다.
8. 안전한 Git 작업: `worktrees/peter_rules`의 `peter`에서만 작업하고 non-force 푸시와 한국어 커밋을 사용한다. 새 브랜치/worktree는 생성하지 않는다. 담당 파일을 나누고 메인이 커밋·푸시를 조율한다. 학습 실행자는 한 번에 하나다.
9. 매 실험 완료 시 본 문서의 **상태·새 증거 링크·주의사항**을 갱신한다. 다른 세션에서 진행 중인 상태는 확인 전 임의로 완료 처리하지 않는다.

## 다음 진입점

- **현재 후속 실행**: 위 후속 절차와 T16/T18의 원장·실행 관리자 확인을 우선한다.
- **과거 설계 열람**: [G1~G5 계획](../experiments/age995/interactions/PLAN.md), [T4 인계 기록](HANDOFF_MODEL_FAMILY_COMPARISON_20261014.md)은 당시 기록이다. 이미 실행한 절차를 재시작하는 지시가 아니다.
- **프로젝트 필수 과제**: [과제 가이드](PROJECT_GUIDE_KO.md)와 팀 합의의 최신본을 확인. XGBoost/LightGBM은 필수 `sklearn GradientBoostingClassifier`를 대체하지 않는다.
