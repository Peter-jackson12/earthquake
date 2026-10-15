# Peter · age995 후속 실험

브랜치: `peter-exp/age995-outlier-treatment`. 기존 `peter`와 충돌하는 `peter/exp/...` 대신 만든 개인 실험 브랜치다.
[실행 전 계획](PLAN.md) · [참고자료 4건](../../docs/AGE_995_REFERENCE_NOTES_KO.md) · [1차 결과](../../reports/AGE995_EXPERIMENT_20261013.md)

A=원값 유지, B=원값+표시, C=학습 Fold 중앙값+표시. 모델·Fold를 고정한 첫 비교이며 광범위한 이상치 삭제는 하지 않았다. 이전 최종 모델과 별도의 개발 CV 연구다. 홀드아웃 평가 명령은 없다.

## 실행

저장소 루트에서 requirements.txt를 설치하고, 기존 v1.0.1의 development.csv와 split_manifest.csv만 data/에 둔다. 데이터 해시가 다르면 거부한다. 출력 디렉터리는 새 경로를 사용한다.

```bash
python -m pip install -r requirements.txt
python experiments/age995/run.py --data-dir data --output outputs/age995_abc_v1
```

위 기본 명령은 A/B/C 모두 학습한다(15 Fold). 이미 신뢰할 수 있는 동일 LR C=1 실행의 원시 OOF·metadata·manifest가 있으면 A만 재사용할 수 있다. 이번 실행은 다음 경로로 A를 재사용하고 B/C 10 Fold만 새로 학습했다.

```bash
python experiments/age995/run.py --data-dir data --output outputs/age995_abc_v1 --baseline-run outputs/personal_completion_v1/lr_c1_confirmation
python experiments/age995/audit.py --data-dir data --output outputs/age995_abc_v1 --receipt outputs/age995_abc_v1/independent_audit.json
python -m unittest discover -s tests -p test_age995.py -v
```

재사용할 원본 실행이 없으면 --baseline-run을 생략한다. 메타데이터나 OOF의 조건 검사가 실패하면 점수만 재사용하지 않는다. 이미 존재하는 실험 출력을 덮어쓰지 않는다.

## 공개/비공개

run.py의 모든 행별 예측과 실행 manifest는 outputs/에만 저장한다. 공개할 파일은 검증한 reports/age995_*_summary.csv 및 집계 보고서뿐이다. 원본 CSV·행별 라벨/예측·모델은 public Git에 올리지 않는다.

파이프라인은 원본 38열을 받는다. B/C에서 age_is_995 1개를 내부 생성하므로 사용자 입력에 표시 열을 미리 넣지 않는다. age=0은 보존하고 C의 중앙값에는 포함한다. 현재 app.py와 기존 개인 최종 평가 문서는 이전 작업이며 이 새 연구의 최종 평가가 아니다.
