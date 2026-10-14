# T6 비공개 실행 증거 회수 재점검 — 성능 감사 차단

2026-10-14 KST · `Peter-jackson12/earthquake` · `peter-exp/geo-ablation-v1`

## 판정

**현재 접근 가능한 위치에서 원래 T6 학습 결과와 보완 실행 증거를 회수하지 못했다. T6 성능 비교·실행 원장·저장 모델 독립 감사는 미완료다.** 이번 작업의 실데이터 분류기 학습은 0회, T6 모델 재로딩은 0개다. 과거 완료 횟수는 0이 아닌 **독립 검증 불가(null)**로 유지한다. R12/R1/R0의 성능, 순위, C 대비 차이와 지역·age995 손익을 추정하지 않는다.

이 보고서는 [이전 복구 보고서](GEO_ABLATION_RECOVERY_20261014.md)에 더한 **로컬·이전 대화·Library 회수 범위 기록**이다. 기존 보고서의 C 재감사 PASS는 당시 기록으로 보존하며, 이번에 다시 수행했다고 주장하지 않는다. [이번 회수 집계 JSON](geo_ablation_retrieval_20261014.json)과 [후속 실행 인계문](../experiments/geo_ablation/RECOVERY_HANDOFF_20261014.md)을 함께 읽는다.

## 최신 원격과 기존 작업 보존

시작 시 실제 `git ls-remote`로 확인했다. 조회값은 시점 스냅샷이며 다른 세션의 이후 변경을 막는 고정값이 아니다.

| 브랜치 | 시작 시 원격 HEAD |
|---|---|
| T6 `peter-exp/geo-ablation-v1` | `bb0e652947ebf6ab8f8420f819a9f69d81b18265` |
| T5 `peter-exp/geo-group-cv-v1` | `acd5ecdeefeff619c2fae9639f1627638de59383` |
| T3 `peter-exp/age995-outlier-treatment` | `a4fc5c82a1907d8fc975f0971deaed8d26a4461f` |
| T4 `peter-exp/model-family-comparison-v1` | `a5d9ed563fc2c1cfdfd7e9db527a119d976b4231` |
| `peter` | `29ccab8af428fd074e7642a1524db6d5974b3f4e` |
| `main` | `f58e1200ccd7132c5f8da788217bda70b9b70dd2` |

원래 로컬 체크아웃은 T6 `4377767d51e195045e824d681610495c024ece7d`였고, 조회 당시 작업 트리는 깨끗했다. 새 복사본을 내려받아 작업했으며 기존 체크아웃의 브랜치·파일·출력은 변경하지 않았다. 원래 코드 기준은 `87b8a1d5d0e7918794f75bbd7e9a5e8418295574`, 보완 실행기 기준은 `20e5011cf2cf942e90b7802a51e5ae8096515277`이다.

`preprocess.py`, 데이터 계약, C/G2 실행기, T5 실행·감사기와 T6 실행·감사·설정 파일은 원래 커밋의 Git blob 바이트와 일치했다. `audit_available.py`·`complete_unattempted.py`는 보완 커밋과, `recovery.py`는 시작 HEAD와 대조했다. 총 12개 보호 파일의 실제 SHA와 바이트 일치는 JSON에 기록했다. **비공개 design/seal을 회수하지 못했으므로 이것을 실행 봉인 검증으로 승격하지 않는다.**

## 회수한 것과 회수하지 못한 것

| 확인 위치 | 관찰 | 한계 |
|---|---|---|
| 원래 `지역 변수 제거 연구` 대화 | 대화 조회에는 첨부가 반환되지 않았다. 브라우저에서는 기존 복구 보고서·복구 코드 ZIP·C 기준선 ZIP 링크가 보였다. | 원래 T6 모델/OOF ZIP 링크는 확인되지 않았다. |
| 동명 `지역 변수 제거 연구` 대화 | 인계문과 첨부 없음이 반환됐다. | 실행 완료·보완 완료 증거 없음. |
| 로컬 에이전트 `재현 실험과 독립 검증` | T1~T4 별도 재현 자료와 기존 체크아웃 전환 기록을 확인했다. | 해당 재현 자료를 T6 결과로 사용하지 않았다. |
| 로컬 파일명 검색 | Projects, Documents, Downloads, OneDrive, PycharmProjects, 에이전트 첨부·프로젝트 폴더, 임시 폴더에서 T6 원래/보완 산출물을 찾지 못했다. Earthquake 작업 폴더 내부도 확인했다. | 검색 범위 밖 또는 이름이 다른 보관본의 부재까지 증명하지 않는다. |
| 접근 가능한 G 드라이브 | earthquake/ablation/design_seal/execution_interruption/supplement_seal 파일명 검색에 일치 항목이 없었다. | 서버 전체 Drive 목록 검사가 아니다. |
| ChatGPT Library/Space `ablation` 검색 | `GEO_ABLATION_RECOVERY_20261014.md`만 확인했다. | 기존 복구 보고서이며 원래 결과가 아니다. |
| Library/Space `T6` 검색 | `T6_RECOVERY_DELIVERY_20261014.json`, `earthquake_T6_recovery_code_evidence_20261014.zip`이 관련 결과로 보였다. | 복구 패키지의 파일명만으로 원래 T6 실행 증거를 확보했다고 판단하지 않았다. |
| Library/Space `unattempted` 검색 | 검색 결과 없음. | 보완 미시도를 의미하지 않는다. |

Library 최근 목록에는 `earthquake_C_baseline_recovery_private_20261014.zip`과 T5 비공개·코드 ZIP도 보였다. 이름과 설명은 회수 후보를 식별하는 근거이며 파일 내용·SHA 검증을 대신하지 않는다.

복구 코드 ZIP의 다운로드 완료 이벤트가 시간 내 반환되지 않았고, 완료된 이름의 파일도 확인되지 않았다. 다운로드 상태 페이지는 브라우저 보안 정책으로 접근이 차단됐다. 우회하지 않았으며 **패키지 회수·압축 내용·ZIP SHA 검증을 완료했다고 보고하지 않는다.** 부분 다운로드를 완료된 ZIP으로 사용하지 않았다. 이 접근 문제와 별도로, 원래 T6 결과 패키지는 검색 결과에서 확인되지 않았다.

## 실제 학습 시도 원장 상태

[실행 중단 기록](../experiments/geo_ablation/EXECUTION_DEVIATION.md)은 43시도/42완료/1중단/2미시도를 기록한다. [보완 계획](../experiments/geo_ablation/RESUME_UNATTEMPTED.md)은 Stratified/R12 Fold3·4만 대상으로 한다. 문서와 코드의 존재를 실행 성공으로 해석하지 않는다.

| 항목 | 문서의 기록 또는 조건 | 이번 독립 확인 |
|---|---|---|
| 원래 Stratified/R12 Fold0·1 | 완료·저장 | 원장·모델·OOF 미회수, 검증 불가 |
| 원래 Stratified/R12 Fold2 | 도구 제한으로 중단, 재시도 금지 | 중단 원시 기록 미회수 |
| 원래 Stratified/R12 Fold3·4 | 당시 미시도 | 이후 보완 실행 여부 미확인 |
| 나머지 8개 구성 | 각 5Fold 완료를 요구하는 가용 증거 감사 | 40개 모델·OOF 미회수, 검증 불가 |
| 보완 Fold3·4 | 성공하면 추가 2시도/2완료 | progress·seal·모델·audit 미회수 |
| 조건부 합계 | 45시도/44완료/1중단/0미시도 | 조건부 수치이며 확인된 실적 아님 |

보완이 성공했더라도 Fold2가 중단 상태이므로 **Stratified/R12의 전체 5Fold 평균·SD·pooled OOF는 미완료**다. 4Fold를 5Fold 결과로 표시하지 않는다. 완료 Fold와 중단 Fold2는 재학습하지 않는다. 파일 부재는 재학습 허가나 미시도 증거가 아니다.

## 이번 검증 범위

로컬의 **개발 CSV와 기존 랜덤 split manifest만** 해시 검사했고 각각 계약과 일치했다.

| 파일 | 실제 SHA-256 |
|---|---|
| development.csv | `8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1` |
| split_manifest.csv | `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721` |

두 Group CV manifest, T6 design/seal, metadata와 artifact SHA는 실제 파일을 회수하지 못해 검증하지 않았다. 기존 C의 지표는 [이전 복구 보고서](GEO_ABLATION_RECOVERY_20261014.md)를 참조하며 이번 실행에서 재산술·재추론하지 않았다. C·R12·R1·R0의 새 전체 비교표를 작성할 근거가 없어 신규 모델 지표와 차이는 JSON에서 null이다.

비공개 데이터를 넣지 않은 새 복사본에서 저장소가 요구하는 `python -m unittest discover -s tests -v`를 실행했다. **총 331개, 통과 302개, 건너뜀 29개, 실패·오류 0개.** 건너뜀은 비공개 데이터 통합 또는 선택적 UI 의존성이 필요한 검사다. 테스트 로그 SHA는 JSON에 기록했다. 기존 합성 테스트의 모델/전처리 fit은 실데이터 연구 학습으로 집계하지 않는다.

T6 실제 저장 모델의 제외 피처 누수·train-only 통계·예측/확률·OOF 정합성·성능 재산술은 **모두 실행 증거 부재로 미검증**이다. 합성 테스트 성공은 이를 대신하지 않는다. 홀드아웃 내용 접근·추출·재평가, T1~T5 학습/모델 재평가, T4 통합, 최종 후보 변경은 하지 않았다.

## 차단 지점과 다음 행동

차단 원인은 개발 데이터 부재가 아니라 **원래 T6 실행 증거와 이후 보완 시도 원장의 미회수**다. 이전 런타임의 `/mnt/data`가 이 Windows 세션에 유지된다고 가정하지 않는다.

필요한 것은 원래 `outputs/geo_ablation_v1/` 전체와 `outputs/geo_ablation_unattempted_v1/`의 존재·실행 여부를 밝히는 원시 로그다. 원래42개와 보완2개가 회수되면 SHA와 봉인 출처를 대조한 뒤 원래 가용 증거 감사와 보완 읽기 전용 감사를 별도로 실행한다. 보완 실행기는 학습도 포함하므로 기존 보완 모델을 감사하려고 다시 실행하면 안 된다. 상세 재개 조건·명령·오류 처리·보고 범위는 [후속 인계문](../experiments/geo_ablation/RECOVERY_HANDOFF_20261014.md)에 고정했다.

원래 데이터·OOF·manifest·joblib·원시 로그는 비공개 보관한다. 공개 반영 범위는 이 회수 보고서, 집계 JSON, 후속 인계문과 T6 README의 탐색 링크뿐이다. 이 기록의 Git 반영은 연구 성능 감사 완료를 의미하지 않는다.
