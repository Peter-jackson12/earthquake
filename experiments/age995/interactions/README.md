# C 기준 지역·기초구조·연령 조합 비교

[사전 계획](PLAN.md) · [설정](config.json) · [결과 보고서](../../../reports/AGE995_INTERACTIONS_20261014.md)

원본 38개 입력을 받고 기존 C 전처리를 그대로 유지한 채 연령 구간·지역/기초구조/연령 조합을 희소 원핫 블록으로 추가한다. G1은 연령 구간 대조군, G2/G3/G4는 각각 단일 조합, G5는 세 조합을 함께 넣는다. 모델과 LR 하이퍼파라미터는 고정한다. 이는 개발 연구 코드이며 기존 최종 모델 또는 운영 정책을 교체하지 않는다.

## 준비

저장소 루트와 `requirements.txt` 고정 버전 환경을 사용한다. 실제 학습은 Python3.13.5였다. `data/development.csv`와 `data/split_manifest.csv`는 기존 계약 SHA와 일치해야 한다. 홀드아웃 파일은 필요하지 않다. 보관용 비공개 ZIP을 별도 폴더 `private`에 풀면 `private/age995_abc_v1/C/`와 `private/runs_v1/`가 생긴다. C의 OOF만 옮기지 말고 metadata·manifest·metrics를 포함한 폴더를 그대로 유지한다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_age995_interaction*.py" -v
```

## 이미 생성된 결과 감사 — 재학습 없음

```bash
python -m experiments.age995.interactions.audit --data data --C private/age995_abc_v1/C --runs private/runs_v1 --output outputs/interactions_audit_recheck --models
```

`--models`는 이번에 직접 생성한 25개 Fold 모델을 다시 로드하여 전체 검증 Fold의 예측까지 대조한다. 모델 파일을 보관하지 않았다면 이 옵션을 빼고 OOF·학습 통계·집계 감사를 수행할 수 있다. 감사 기록의 `private_models_repredicted`로 범위를 구분한다. 모르는 출처의 joblib 파일을 넣지 않는다.

출력은 모델/폴드/그룹/지역·구조·연령 프로필/추가 범주 커버리지/오류 전이 CSV, 원시·정규화 혼동행렬, 검사 증거 JSON이다. 감사 출력 경로는 기존 폴더를 덮어쓰지 않는다. 공개 Git에는 주요 모델·Fold·pooled 그룹·혼동행렬과 요약을 보존하며 전체 집계는 증거 ZIP과 감사 출력에 있다.

## 전체 재학습 — 필요한 경우에만

```bash
python -m experiments.age995.interactions.run --data data --C private/age995_abc_v1/C --output outputs/interactions_rerun --code-commit 109abd61e3ebf732e6f3be12b2544f7008f57c6c
```

이 명령은 기존25개 모델을 검증하는 명령이 아니라 **신규25회 학습**이다. 이미 완료된 결과를 확인할 목적이면 위 감사 명령만 사용한다. `--variants G1 G2`처럼 미실행 설정을 제한할 수 있다. 설정별 출력 폴더가 이미 있으면 거부한다.

위 SHA는 실제 학습 실행기/설정의 고정 소스 커밋이다. 실행기·설정 바이트가 바뀌었다면 새로운 실험 ID와 실제 커밋을 지정한다. 저장 모델은 이 모듈과 고정 `config.json`을 함께 필요로 하므로 단일파일 독립 배포 모델로 간주하지 않는다. 기존 `RESEARCH_CANDIDATE.json`·최종 모델·앱을 자동 변경하지 않는다.

## 경계

추가 교차항은 모든 지역에 동일하게 적용한다. 특정 지역의 예측 등급 덮어쓰기, 정답을 이용한 인코딩, 희귀·오류 행 삭제, 클래스 가중치 변경, 홀드아웃 평가를 하지 않는다. C의 수치 연령은 학습Fold 비995 중앙값으로 대체하지만 추가 연령 구간은 원값에서 생성하여995 상태를 구분한다.

기존 개발 OOF를 보고 설계한 탐색적 후속 비교다. CV 개선을 새 독립 최종성능이나 운영 안전 인증으로 부르지 않는다. 전체 점수와3→1,995집단,진단 지역의 손익을 함께 검토한다.
