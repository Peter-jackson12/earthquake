# Peter · age995 이상치 처리 실험 브랜치

**현재 브랜치: `peter-exp/age995-outlier-treatment`**
기존 `peter`와 `main`을 변경하지 않는 개인 후속 연구다. 요청한 `peter/exp/...` 이름은 기존 `peter` 브랜치와 충돌하여 별도 이름을 사용했다.

**첫 비교 완료:** LR·기존 5-Fold를 고정하여 원값 유지(A), 표시 추가(B), 학습 Fold 중앙값+표시(C)를 비교했다. C의 개발 CV Macro F1은 **0.688687**이다. 홀드아웃 재평가나 이전 최종 모델 교체는 하지 않았다.

- [이번 실험 결과와 집단별 손익](reports/AGE995_EXPERIMENT_20261013.md)
- [실행 방법과 코드](experiments/age995/README.md)
- [실행 전 계획](experiments/age995/PLAN.md)
- [저장된 참고자료 4건](docs/AGE_995_REFERENCE_NOTES_KO.md)

기존 개인 분석의 README 원문은 [README_PERSONAL.md](README_PERSONAL.md)에 보존되어 있다. `app.py`와 이전 최종 평가 문서는 기존 분석이며 이번 age995 연구의 최종 평가가 아니다. 새 결과는 `reports/age995_*`와 별도 보고서로 분리했다.
