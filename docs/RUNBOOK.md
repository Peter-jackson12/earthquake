# earthquake

네팔 건물 지진 피해 등급(1/2/3) 예측 팀 프로젝트.
**전처리 v1.0.1 + LR / DT / RF / GradientBoostingClassifier 공통 기준선 실행기**를 제공합니다.

## 평가 계약

주지표는 **Macro F1**. 개발 208,480건, 최종 홀드아웃 52,121건을 분리한 기존 80/20 분할과 개발 데이터의 고정 5-Fold를 사용합니다.
이 분할은 작업 기본안입니다. 공식 과제 규정·팀 합의가 별도로 확인되면 우선 적용하며, 바뀐 실험은 새 버전으로 구분합니다.
원본 38개 입력, ID·정답 제외, 학습 fold 안에서만 인코더·LR 수치 스케일러 fit, 희소 행렬 유지가 원칙입니다.
홀드아웃은 기준선·튜닝에 사용하지 않습니다. 현재 모델이 미래 지진이나 미관측 지역에서 검증됐다는 뜻도 아닙니다.

## 시작하기

검증 환경은 Python 3.13.5입니다. 먼저 Python 3.13 가상환경에서:

```bash
python -m pip install -r requirements.txt
```

이미 공유받은 전처리 v1.0.1 패키지의 `data/` 폴더를 이 저장소 루트에 복사하세요.
원본 CSV만 있다면 저장소 **밖**에 둔 후 다음 명령으로 재생성할 수 있습니다. 비어 있지 않은 data 폴더는 기본적으로 덮어쓰지 않습니다.

```bash
python preprocess.py --input ../nepal_building_damage.csv --output data
python -m unittest discover -s tests -v
python baseline.py --rows 2000 --output outputs/preflight
python baseline.py --output outputs/baseline_full --jobs 2
```

첫 학습 명령은 2,000건의 **실행 확인용 표본**, 두 번째는 개발 전체 **5-Fold 평가**입니다. 둘을 같은 성능표에 섞지 마세요.
`--models lr dt rf gbc`처럼 일부 모델만 선택할 수 있으며, 같은 결과 경로를 재사용하면 오류로 멈춥니다. 새 실험은 새 출력 폴더를 사용하세요.
`python` 명령은 Windows에서도 동일하며, 가상환경 활성화 방식만 환경에 맞추면 됩니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `preprocess.py` | 원본과 동일한 v1.0.1 전처리·분할·검증 |
| `baseline.py` | Dummy + 필수 4개 모델의 fold 내부 학습·평가 |
| `configs/baseline.json` | 성능 확인 전에 정한 초기 하이퍼파라미터 |
| `configs/data_contract.json` | 승인된 개발 CSV·분할 파일의 해시 |
| `tests/` | 합성 검사 + 데이터 설치 시 실행되는 통합 검사 |
| `docs/PLAN.md` | 완료 조건과 후속 실험 순서 |
| `AGENTS.md` | 팀원·코딩 에이전트의 공통 작업 규칙 |
| `reports/` | 원시 행 없는 검증·실험 집계 |

## 초기 모델 설정의 의미

LR만 수치 5개를 표준화합니다. DT는 depth=12, RF는 100 trees / depth=20 / leaf=2, GBC는 100 stages / depth=3 / **max_features=sqrt**로 시작합니다.
이는 고정한 첫 비교 설정이며 모델별 최적 성능이나 모두 라이브러리 기본값을 뜻하지 않습니다.
특히 GBC의 sqrt는 계산량을 제한한 설정입니다. 고차원 원핫에서 RF·GBC가 약하면 깊이·후보 변수 수에 의한 과소적합 가능성을 별도 실험으로 확인해야 합니다.
클래스 가중치, SMOTE, 파생변수, 자동 튜닝, 최종 홀드아웃 평가는 이 기준선에 없습니다.

## 결과와 검증

실행 폴더에 `run_metadata.json`, 모델별 `fold_metrics.json` / `metrics.json`, OOF 예측, 첫 fold 모델이 저장됩니다.
각 건물의 OOF 예측이 정확히 한 번인지, 확률 열 순서가 1/2/3인지, 첫 fold 모델을 다시 불러와 예측이 같은지 검사합니다.
Macro F1의 fold 평균과 전체 OOF Macro F1은 다른 집계이므로 구분합니다. fold 표준편차는 신뢰구간이 아닙니다.
수렴 경고도 결과에 남기며 숨기지 않습니다. 모델 파일은 신뢰할 수 있는 본인 실행에서 생성된 것만 여세요.

CI에서는 원본 데이터 없이 합성 검사를 수행합니다. 데이터 의존 통합 검사 19개는 명시적으로 건너뛰며, 이를 실제 데이터 재현 완료로 세지 않습니다.
데이터를 설치한 환경에서는 전체 테스트를 실행할 수 있습니다.

## 공개 범위

이 저장소는 코드·설정·검사·집계 결과만 공개합니다. `data/`, 원본 CSV, 개별 건물 예측·라벨, 모델, 기존 ZIP·원시 행 포함 노트북은 업로드하지 않습니다.
공통 `split_manifest.csv`도 팀의 별도 데이터 공유 경로로 전달하고, 공개 저장소에는 생성 규칙과 해시만 보존합니다. 초대받은 팀원도 GitHub에서 데이터를 자동으로 받는 것은 아닙니다.
`.gitignore`는 실수 방지 장치이며 `git add -f`까지 차단하지 않습니다. 커밋 전 목록을 검토하세요.
