# age995 후속 실험 D/E/F

결과: [완료 보고서](../../reports/AGE995_FOLLOWUP_20261013.md)

실행 전 동결 규칙: [FOLLOWUP_PLAN.md](FOLLOWUP_PLAN.md), [followup_v1.json](followup_v1.json)

## 상태

- D/E1/E2/E3/F1/F2, 총 30개 실제 개발 Fold 학습 완료.
- A/B/C 재학습 없이 기존 625,440개 OOF를 재사용하고 독립 검증.
- 총 9개 설정·45개 Fold·1,876,320개 OOF·162개 그룹 혼동행렬 감사 PASS.
- 기존 C를 개발 비교 기준으로 유지. E2는 근접 대안이며 우열을 확정하지 않음.
- 원본·전처리 v1.0.1·A/B/C·main/peter 보존. 최종 홀드아웃 재평가 없음.

## 파일 역할

`followup.py`는 고정 설정을 실행한다. 원본 정수 검증 후 별도 복사본에서 상한절단하며, 삭제는 학습 Fold에만 적용한다. 원핫과 스케일러는 삭제 후 남은 학습 행에서만 fit한다.

`audit_followup.py`는 학습·평가 헬퍼를 호출하지 않고 OOF를 ID로 대조한 뒤 pandas crosstab과 직접 P/R/F1 산술로 검증한다. 학습 삭제 마스크·분위수·등급·지역·희귀 범주·검증 분모·OOF checksum도 독립 대조한다. 훈련 모델 재학습이나 계수 감사는 수행하지 않는다.

`tests/test_age995_followup.py`와 `tests/test_age995_followup_audit.py`는 48개 합성/손상 거부 테스트다. 합성 점수는 실데이터 성능이 아니다.

공개 결과는 `reports/age995_followup_v1/`에 있다. `pooled_group_summary.csv`는 전체/원본995/비995의 등급별 P/R/F1·과소예측 비율을 담고, `pooled_confusion_matrices.jsonl`은 원시·행 정규화 행렬을 담는다. 전체 Fold·지역·범주 집계는 아래 감사 명령과 별도 전체 증거 패키지에서 재현/확인한다.

## 데이터 배치

공개 GitHub에는 원본 CSV, 준비 데이터, 행별 OOF 또는 모델이 없다. 기존 비공개 파일을 다음 경로에 둔다.

```text
data/development.csv
data/split_manifest.csv
age995_abc_v1/A/lr/oof_predictions.csv.gz
age995_abc_v1/B/lr/oof_predictions.csv.gz
age995_abc_v1/C/lr/oof_predictions.csv.gz
```

A/B/C는 OOF만 복사하지 말고 기존 `run_metadata.json`, `run_manifest.csv`, `lr/metrics.json`, `lr/fold_metrics.json`을 포함한 전체 설정 폴더를 유지한다. `configs/data_contract.json`의 SHA와 일치해야 한다. 홀드아웃 feature/label 파일은 이번 재현에 필요하지 않다.

## 재현 명령

저장소 루트, requirements.txt의 고정 버전 환경에서 실행한다. 전체 실데이터 실험은 Python 3.13.5에서 수행했다. 아래 출력 폴더가 이미 존재하면 덮어쓰지 않고 오류로 종료하므로 새 경로를 사용한다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_age995*.py" -v
python -m experiments.age995.followup --data-dir data --output outputs/age995_followup_rerun --variants D E1 E2 E3 --code-commit 9b89cee66514b91c5e306153611b7818475616c4
python -m experiments.age995.followup --data-dir data --output outputs/age995_followup_rerun --variants F1 F2 --code-commit 9b89cee66514b91c5e306153611b7818475616c4
python -m experiments.age995.audit_followup --data-dir data --abc age995_abc_v1 --new outputs/age995_followup_rerun --output outputs/age995_followup_audit_rerun --variants D E1 E2 E3 F1 F2
```

`--code-commit`은 실행기의 소스 버전이다. 해당 버전의 핵심 소스가 유지된 현재 브랜치에서 실행할 수 있다. 소스를 바꾼 뒤 이 SHA를 그대로 쓰지 말고, 새 규칙/ID를 고정하고 실제 소스 커밋을 기록한다. BLAS/학습 스레드는 실행기 내부에서 1로 제한한다.

이미 생성된 비공개 결과를 재검사할 때는 학습 명령을 생략하고 `--new outputs/age995_followup_v1_20261013`로 감사 명령만 실행한다. 감사 출력 경로도 새로 지정한다. 원래 A/B/C audit.py는 학습행 삭제를 가정하지 않으므로 D/F 검증에 사용하지 않는다.

## 실험 해석 경계

q99/linear는 이 프로젝트의 사전 지정 규칙이며 실제 상한은 5개 Fold 모두90이었다. F1/F2는 기존 문헌 메모에 기록된 임계값을 분리한 실험으로, 논문 전체 파이프라인 재현이 아니다. PDF의 이번 세션 재접근 실패를 보고서에 명시했다.

숫자상 C가 1위지만 E2와 차이는 0.000024에 불과하다. D/F1/F2는995 집단을 모두3등급으로 예측했다. 전체 점수나3등급 Recall 한 가지만으로 정책을 고르지 않는다. 995의 공식 코딩 의미는 미확정이고, 이미 사용한 홀드아웃은 새로운 독립 최종 평가가 아니다.
