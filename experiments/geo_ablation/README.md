# T6 지역 변수 Ablation — 재개 진입점

**현재 상태: `BLOCKED_T6_PRIVATE_EVIDENCE_MISSING` (2026-10-14 KST).**
코드·기존 C 기준선 재검증은 완료했지만, 이전 T6의 비공개 실행 증거를 현재 세션에서 회수하지 못했다. T6 성능 비교 완료나 45회 학습 완료를 뜻하지 않는다.

1. [복구 보고서](../../reports/GEO_ABLATION_RECOVERY_20261014.md)와 [기계 판독 요약](../../reports/geo_ablation_recovery_summary.json)을 먼저 읽는다.
2. 원래 [PLAN](PLAN.md), [실행 중단 기록](EXECUTION_DEVIATION.md), [미시도 Fold 보완 계획](RESUME_UNATTEMPTED.md)을 보존한다. 보완 계획의 존재는 보완 학습 성공의 증거가 아니다.
3. **`outputs/geo_ablation_v1/` 전체와 `outputs/geo_ablation_unattempted_v1/`의 실행 여부·증거를 회수한다.** 원래 design/seal, metadata, folds, OOF, joblib, audit_available 및 보완 audit가 필요하다. 이전 `/mnt/data`가 유지된다고 가정하지 않는다.
4. 회수된 파일의 SHA·시도 원장을 대조한 뒤 원래 독립 감사 경로를 사용한다. 현재 `recovery.py`의 `training_allowed=false`는 파일 부재를 미실행으로 오해하지 않도록 하는 별도 점검 결과다. 기존 실행기를 수정하거나 기술적으로 잠그는 패치는 아니다. **`run fit-all` 또는 `complete_unattempted`를 지금 바로 실행하지 않는다.**

## 변경한 것과 바꾸지 않은 것

새 `recovery.py`와 `tests/test_geo_ablation_recovery.py`는 중단 후 추가한 읽기 전용 복구/검증 코드다. 학습 전 봉인 코드인 것처럼 주장하지 않는다. 기존 C, T1~T5, T6 run/audit/config, 원본 전처리를 변경하지 않는다. C/G2 및 완료된 T6 Fold 재학습도 없다.

R12: geo3 제외 / R1: geo2·geo3 제외 / R0: geo1·geo2·geo3 제외. 원본 38개 입력 스키마와 C 연령 정책은 그대로다. 제거 대상의 원핫 열만 모델 변환에서 빠진다. 그룹 분할 키는 모델 피처가 아니다.

## 분류기 학습 없는 재검증

GitHub 브랜치를 체크아웃하고 비공개 C 복구 패키지를 저장소 루트에 배치한다. 각 출력 경로는 새 경로여야 한다.

```bash
python -m unittest discover -s tests -p "test_geo_ablation_recovery.py" -v
python -m experiments.geo_ablation.recovery --scheme stratified --destination outputs/recheck/C_stratified.json
python -m experiments.geo_ablation.recovery --scheme geo2 --destination outputs/recheck/C_geo2.json
python -m experiments.geo_ablation.recovery --scheme geo1 --destination outputs/recheck/C_geo1.json
```

이 명령은 C OOF와 저장된 Group-CV 모델을 재검증할 뿐 T6 학습이나 완결 감사를 대신하지 않는다. 신뢰할 수 있는 프로젝트 생성 joblib만 고정 SHA 확인 후 로드한다. 홀드아웃 feature/label을 추출하거나 읽지 않는다. 원시 데이터·OOF·manifest·모델을 공개 Git에 추가하지 않는다.
