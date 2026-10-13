# 후속 검증·불균형 비교 실행 안내

범위: 개인 `peter`. 공통 `main`, `preprocess.py` v1.0.1, 기존 데이터·분할·과거 집계는 보존한다.
기존 원본 CSV의 재첨부는 과거 OOF 예측 파일의 복구가 아니다. 새 결과와 기존 결과를 구분한다.

## 설치와 검증

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

원본 CSV는 UTF-8 BOM이 있으며 SHA가 기존 계약과 같아야 한다. v1.0.1 패키지의 `data/`를 별도로 설치한다.
데이터 없는 CI의 skip은 실제 전체 데이터 전처리나 모델 학습을 검증했다는 뜻이 아니다.
새 통합 테스트는 데이터가 있을 때 실제 5개 Fold를 비스케일·스케일 경로로 각각 fit/transform한다.

## 이미 완료한 결과의 확인 — 우선 경로

이번에 전달한 비공개 증거 ZIP의 `outputs/`를 개인 저장소 루트에 놓는다. 과거 결과 디렉터리를 덮어쓰지 않는다.
원시 데이터·OOF·run_manifest·학습 모델은 public Git에 추가하지 않는다. 코드와 검토한 집계만 공개한다.
다음은 모델을 재학습하지 않는 검증이다. 출력 디렉터리는 새 이름을 사용한다.

```bash
python audit_results.py --runs outputs/class_weights_v1 outputs/lr_rus_v1 --output outputs/followup_metrics_recheck
python scripts/audit_eda_oof.py --runs outputs/class_weights_v1 outputs/lr_rus_v1 --output outputs/followup_eda_oof_recheck
```

EDA만 재현할 때:

```bash
python scripts/audit_eda_oof.py --output outputs/eda_recheck
```

`--source`를 추가하면 원본 CSV의 SHA만 확인한다. EDA는 항상 계약이 고정된 개발 CSV로 계산한다.
OOF를 제공하지 않은 EDA 실행의 상태는 `NOT_PROVIDED`이며 OOF 검증 성공이 아니다.
새 감사기는 동일 ID·Fold·정답, 확률 범위·합·열 순서와 예측 일치, 혼동행렬, CV 평균·표준편차, 지역별 일대일 결합을 검사한다.
기존 `audit_results.py`는 등급별 P/R/F1도 별도로 재계산한다. 둘을 다른 새 출력 폴더로 실행한다.

## 학습 명령 — 새로운 실행이 필요한 경우에만

아래 두 설정은 이미 전체 5-Fold를 실행했다. 같은 결과를 확인하기 위해 다시 학습하지 않는다.
사용자 요청, 재현 확인 또는 승인된 조건 변경이 있을 때만 새로운 실험 ID/출력 경로를 사용한다.

```bash
python baseline.py --config configs/class_weights.json --models dt rf gbc --output outputs/class_weights_reproduction --jobs 2
python baseline.py --config configs/lr_rus.json --models lr --output outputs/lr_rus_reproduction --jobs 2
```

DT/RF는 `class_weight=balanced`, GBC는 학습 Fold 라벨만으로 계산한 값을 `model__sample_weight`로 전달한다.
RUS 순서는 **원본 학습 Fold의 스키마 검사·인코딩·스케일링 → RUS → 무가중치 LR**이다.
imblearn 중간 단계의 nested Pipeline 제약 때문에 동일 전처리의 schema/columns 단계만 펼치며 내용은 바꾸지 않는다.
검증 단계에서는 샘플러가 실행되지 않는다. 샘플링 전후 등급 수, 가중치, 경고, 반복 수, OOF SHA를 남긴다.

학습과 함께 Git HEAD/dirty를 기록하되 Git checkout이 아니면 null로 남기고 실제 코드 SHA-256을 사용한다.
이번 실행은 GitHub blob과 대조한 컨테이너 복사본에서 수행했으며 원격 clone 성공으로 기록하지 않는다.
시간은 실행 환경의 참고 값이다. 이전 실행과 장비·동시 부하를 엄밀히 통제한 속도 벤치마크가 아니다.

## 원문 API 참고

- GBC sample_weight: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- RandomUnderSampler: https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html
- 학습 시에만 샘플링하는 Pipeline: https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html

실행 환경은 `imbalanced-learn==0.14.1`로 고정했고 설치된 버전에서 테스트했다. stable 문서의 최신 버전과 구분한다.
