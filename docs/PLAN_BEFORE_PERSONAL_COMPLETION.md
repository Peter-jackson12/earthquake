# 실행 계획과 현재 상태 — 2026-10-13 후속 반영

목표: v1.0.1 기반 기준선 → 불균형·튜닝 비교 → 오류 해석 → 최종 평가·Streamlit 제출물.
공식 요구사항은 [PROJECT_GUIDE_KO.md](PROJECT_GUIDE_KO.md), 이전 기준선은 [BASELINE_REPORT.md](../reports/BASELINE_REPORT.md)를 본다.
최신 개인 결과는 [후속 보고서](../reports/FOLLOWUP_REPORT_20261013.md), 실행은 [후속 Runbook](FOLLOWUP_RUNBOOK_20261013.md)을 우선한다.

> 완료 기록은 peter 개인의 선행 작업이며 팀 전체 완료나 팀원 모두의 수행을 뜻하지 않는다.

| 단계 | 상태 | 근거 / 다음 조건 |
|---|---|---|
| 기존 전처리 인계 | 보존·재검증 완료 | v1.0.1 및 기존 data/Manifest 해시 불변 |
| 이전 기준선·후보 변수 수·LR balanced | 8개 설정·40 Fold 기존 기록 보존 | 기존 summary CSV; 새로 재학습하지 않음 |
| 개발 데이터 EDA | 개인 1차 해석·재현 코드 완료 | scripts/audit_eda_oof.py, eda_summary.csv 55개 집계 |
| 과거 행별 OOF 독립 재검증 | 일부 미완료 | 과거 8개 원시 OOF 없음. 전역 집계만 산술 확인 |
| 신규 가중치·RUS 비교 | 4개 설정·20 Fold 완료 | DT/RF balanced, GBC sample_weight, LR RUS |
| 신규 OOF·지역 오류 | 행별 재검증 완료 | 4개×208,480 예측; followup_verification.json |
| 테스트 | 실제 data 53 통과 | data 없음: 31 통과 / 22 skip; 원격 CI는 별도 확인 |
| 팀 문제 정의·공통 분할 합의 | 미확정 | 80/20·seed42·5-Fold를 팀 확정으로 취급하지 않음 |
| SMOTE 계열 비교 | 설계 검토만, 실행 미완료 | 원본 정수 계약·명목형·특수값·지역 조합 안전성 파일럿 필요 |
| 제한 CV 튜닝 | 미실행 | NEXT_EXPERIMENTS에서 중복된 C=1/RF 조합 제외 |
| 최종 모델·홀드아웃 평가 | 미실행 | 후보와 평가 계획 확정 후 최종 평가 |
| Streamlit·README·팀 Notion·제출 | 진행 필요 | 실제 집계 사용, main 채택·개인 수행 기록 별도 |

## 다음 우선순위

팀 기준 합의 및 과거 OOF 확보 → 기존 결과 재사용 → 제한 튜닝·SMOTENC 파일럿 → 최종 후보와 평가 → 대시보드·제출물 통합.
무가중치 LR은 현재 CV 최고 후보이나 최종 모델이 아니다. 가중치 RF는 Macro F1과 3→1 악화를 함께 비교한다.
[미완료 실험](NEXT_EXPERIMENTS_20261013.md)에 완료/미완료와 중복 제외 기준을 남겼다.
[논문 설명 보완](PAPER_BENCHMARK_CORRECTIONS_KO.md)은 기존 문헌 메모와 함께 읽는다.

## 변경·공유 규칙

팀원마다 전체 분석·모델링 과정을 수행한다. 개인 결과를 팀 전체 수행으로 대체하지 않는다.
main에는 팀이 검토·재현해 채택한 변경만 최신 main에서 별도 PR로 반영한다. peter 전체를 무조건 병합하지 않는다.
전처리·분할·과거 결과를 덮어쓰지 않는다. 원본·행별 예측·모델·비밀 값은 공개하지 않는다.
최종 평가 전 승인, 공유·권한 경계는 AGENTS.md를 따른다. 예약·무인 지속 실행을 시작하지 않는다.
