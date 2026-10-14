# age995 결론 후속 작업 — C 기준 고정 및 오류 진단

2026-10-14. 대상: `peter-exp/age995-outlier-treatment`. 출발 HEAD: `0de4268f55e38368f25a0a2cc9c6cd3f77266be0`.

## 이번 단계

새 전처리를 더 탐색하기 전에 기존 C를 개인 후속 연구의 기준으로 기록하고, 저장된 A/B/C/D/E/F OOF에서 예측 교체·확률·학습 연령 범위·지역/구조별 과소예측을 분석했다. **실데이터 모델 재학습, 신규 모델 학습, 임계값 튜닝, 홀드아웃 평가를 하지 않았다.** 기존 최종 모델이나 main/peter의 정책을 교체하지 않는다.

`RESEARCH_CANDIDATE.json`은 연구 결정 기록이다. 실제 파이프라인은 기존 `experiments.age995.run.make_pipeline('C')`를 그대로 사용한다. 이 JSON을 기존 앱이 자동으로 읽도록 바꾸지 않았으며 배포 설정이 아니다.

`diagnostics_v1.json`은 결과를 이미 본 개발 OOF의 **사후 기술적 진단** 설정이다. 80% 확률·0.1 확률 차이는 설명용 구간이며 실제 의사결정 임계값이 아니다. 비용 가중치도 실제 안전·경제 비용으로 합의한 값이 아니다. 새 실험 사전등록이나 통계적 유의성 검증이라고 부르지 않는다.

## 재현

기존 비공개 OOF ZIP을 `private_evidence/`에 푼다. 원래 A/B/C 및 D/E/F 설정 폴더와 메타데이터/manifest를 유지한다. 공개 저장소에는 이 데이터가 없다. 개발 CSV와 split manifest는 기존 계약 SHA와 같아야 한다. 홀드아웃 feature/label은 필요하지 않다.

```bash
python -m unittest discover -s tests -p "test_age995*.py" -v
python -m experiments.age995.diagnostics --data-dir data --private-root private_evidence --output outputs/age995_diagnostics_20261014
python -m experiments.age995.audit_diagnostics --data-dir data --private-root private_evidence --report-dir outputs/age995_diagnostics_20261014
```

출력 경로가 존재하면 분석기는 덮어쓰지 않는다. 코드/데이터/OOF/실행 metadata SHA와 ID·정답·Fold·확률을 대조한 뒤 집계한다. 감사기는 분석 함수를 재호출하지 않고 sklearn F1과 별도 3×3×3 전이 집계로 다시 계산한다.

분석 출력: `group_diagnostics.csv`, `paired_comparisons.csv`, `paired_transitions.csv`, `cost_sensitivity.csv`, `error_profiles.csv`, `training_age_support.csv`, `age_affine_identity.csv`, `verification.json`. 공개 Git에는 요약과 결과 보고서를 넣고, 전체 집계/로그는 전달 ZIP 및 위 재현 명령으로 보존한다. 원본 CSV·행별 예측·라벨·모델은 공개하지 않는다.

지역·구조별 차이는 교란과 집단 크기의 영향을 받을 수 있다. 3→1/3→2 비율의 분모는 **해당 집단의 실제 3등급**이고, 전체 오류 기여율의 분모는 전체 3→1 오류다. 두 비율을 혼동하지 않는다. 코드형 범주를 실제 자재/지명으로 임의 번역하지 않는다. 집단별 실제 3등급이 30건 미만이면 작은 표본 경고를 기록한다.

## 이어지는 작업의 기준

995 대체값의 미세 조정을 우선순위에서 내린다. C 기준으로 비995·지역·구조의 심각한 과소예측을 개선할 단일 변경 실험을 설계하되, 관측된 고위험 지역을 예측 등급 강제 변경이나 행 삭제 규칙으로 사용하지 않는다. 새 후보를 실제 안전 정책으로 채택하기 전에는 3→1·3→2·과대예측의 비용과 별도 독립 평가를 정해야 한다. 기존 홀드아웃은 다시 선택에 사용하지 않는다.
