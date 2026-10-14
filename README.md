# Peter · age995 이상치 처리 실험 브랜치

## 세션 인계 및 후속 연구 시작점

- **[현재 개인 연구 현황 · Research Status](docs/RESEARCH_STATUS.md)** — A~F 전처리·C 연구 기준·오류 진단·병렬 트랙 진행 상태, 검증 규약과 인계 체크리스트.
- **[새 세션 인계: XGBoost·LightGBM vs LR 모델 계열 비교](docs/HANDOFF_MODEL_FAMILY_COMPARISON_20261014.md)** — 다른 모델 계열 연구(T4) 시작 지시서. 여기서는 아직 신규 XGBoost/LightGBM을 학습하지 않았음.
- **[다른 세션에서 진행 중인 G1~G5 특성 조합(T3)](experiments/age995/interactions/PLAN.md)** — LR+C 기반 지역·구조·연령 상호작용 실험. 모델 계열 비교와 구분.

새 세션은 **최신 원격 HEAD → 연구 현황 → 해당 트랙의 코드·보고서·원시 OOF 검증** 순서로 확인한다. 연구 상태 문서는 작성 시점의 스냅샷이므로, 다른 세션의 후속 커밋을 반드시 재확인한다.


**현재 브랜치: `peter-exp/age995-outlier-treatment`**
기존 `peter`와 `main`을 변경하지 않는 개인 후속 연구다. 요청한 `peter/exp/...` 이름은 기존 `peter` 브랜치와 충돌하여 별도 이름을 사용했다.

**첫 비교 완료:** LR·기존 5-Fold를 고정하여 원값 유지(A), 표시 추가(B), 학습 Fold 중앙값+표시(C)를 비교했다. C의 개발 CV Macro F1은 **0.688687**이다. 홀드아웃 재평가나 이전 최종 모델 교체는 하지 않았다.

- [이번 실험 결과와 집단별 손익](reports/AGE995_EXPERIMENT_20261013.md)
- [실행 방법과 코드](experiments/age995/README.md)
- [실행 전 계획](experiments/age995/PLAN.md)
- [저장된 참고자료 4건](docs/AGE_995_REFERENCE_NOTES_KO.md)

기존 개인 분석의 README 원문은 [README_PERSONAL.md](README_PERSONAL.md)에 보존되어 있다. `app.py`와 이전 최종 평가 문서는 기존 분석이며 이번 age995 연구의 최종 평가가 아니다. 새 결과는 `reports/age995_*`와 별도 보고서로 분리했다.
