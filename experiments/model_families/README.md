# T4 — XGBoost/LightGBM A/C 비교

학습 전 계획은 [PLAN.md](PLAN.md), 고정 설정은 [config.json](config.json).
별도 브랜치 `peter-exp/model-family-comparison-v1`이며 T3 G1~G5와 결합하지 않는다.

## 환경과 비공개 입력

Python 3.13.5에서 다음을 실행한다. 정확한 실행 환경은 결과 verification.json에도 기록한다.

```bash
python -m pip install -r experiments/model_families/requirements.txt
```

기존 전처리 v1.0.1의 `development.csv`, `split_manifest.csv`만 사용한다.
CSV를 다시 만들지 않고 데이터 계약 SHA와 대조한다. 원본/홀드아웃 feature·label은 필요하지 않다.
`PRIVATE_LR_ROOT`는 기존 비공개 `age995_abc_v1`의 A/C 하위 폴더를 가진 경로다.
각 폴더의 `run_metadata.json`, `run_manifest.csv`, `lr/metrics.json`, `lr/oof_predictions.csv.gz`가 필요하다.
A/C의 OOF SHA를 코드에 고정하고 메타데이터/분할/정답/확률/지표를 검사한 뒤 재사용한다.

## 재현 순서

다음 경로는 사용자 환경에 맞는 **새 빈 비공개 경로**로 지정한다. 새 결과를 기존 연구에 덮어쓰지 않는다.

```bash
python -m experiments.model_families.run \
  --data-dir data \
  --lr-root PRIVATE_LR_ROOT \
  --output PRIVATE_NEW_RUN

python -m experiments.model_families.audit \
  --data-dir data \
  --root PRIVATE_NEW_RUN \
  --lr-root PRIVATE_LR_ROOT \
  --output NEW_AGGREGATE_OUTPUT

python -m unittest discover -s tests -p "test_model_families*.py" -v
python -m unittest discover -s tests -p "test_cv_fold_validation.py" -v
```

PowerShell에서는 줄 끝 `\` 대신 한 줄 명령을 사용하거나 PowerShell의 줄 연속 문법을 쓴다.
실험 실행기의 모델 스레드는2, Fold/모델 실행은 순차다. BLAS의 불필요한 외부 병렬화도 제한한다.
새 모델 총20fit, LR 재학습0회, Optuna/조기종료/클래스가중치/샘플링0회다.
`TRAINED_PENDING_INDEPENDENT_AUDIT`는 학습만 끝난 상태이며, 별도 감사의 `PASS`가 있어야 검증 완료다.

## 감사·공개 범위

학습기는 sklearn, 감사기는 crosstab/직접 수식으로 지표를 계산한다.
모든 예측 행의 ID/Fold/정답/확률/해시와 등급별 지표·분모를 검사하고 저장된20모델을 전체 검증Fold에서 다시 추론한다.
과거 LR는 OOF와 메타데이터를 재검증하되 모델을 재학습하지 않는다.
감사에서 joblib 파일은 **이 실행기가 만든, SHA가 맞는 신뢰 가능한 전처리기만** 읽는다.

PRIVATE_NEW_RUN에는 건물별 정답·확률·모델이 들어 있다. **공개 저장소에 추가하지 않는다.**
공개 대상은 코드·설정·집계·감사 요약·보고서뿐이다. 모델 파일이 `.txt`여도 비공개다.
각 집계의 support와 오류분모를 유지하며, 지역·구조·연령 절단면은 서로 중복되므로 원인 기여율로 더하지 않는다.
995가 공식 결측 코드라는 가정은 하지 않는다. C는 LR 연구 기준이며 트리에 자동 채택하지 않는다.

## 해석 경계

이 v1은 같은 LR식 표준화/OHE 희소 표현에서 미튜닝 고정 설정을 비교한다.
300 boosting rounds는 두 구현의 같은 표현력이나 같은 실행시간을 의미하지 않는다.
XGB의 희소 미저장값 처리와 LightGBM `zero_as_missing=false`의 의미도 다르다.
성능 차이를 순수한 트리 성장 방식만의 효과로 해석하지 않는다.

새 모델 점수가 낮아도 라이브러리 전체의 성능 한계를 결론내리지 않는다.
반대로 점수가 높아도 기존 홀드아웃 재평가·팀 최종 모델/앱 교체·main 병합은 하지 않는다.
모델별 native categorical이나 더 큰 학습 예산은 별도 버전의 사전 계획으로 분리한다.

실행용 기반 소스는 GitHub의 base commit blob과 일치하는 Library 보관본으로 복구했다.
현재 환경에서 git clone은 DNS 실패했으며, 성공했다고 표시하지 않는다.
`inherited_sources.json` 및 결과 source SHA는 이 복구의 검증 근거다.
