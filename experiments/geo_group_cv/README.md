# T5 지역 Group CV — 완료·인계 진입점

**실제 신규20회 학습 + 독립 OOF/저장 모델 감사 PASS.**
이 트랙은 `peter-exp/geo-group-cv-v1`에서만 수행했다. T3/T4 또는 기존 개인 최종 모델을 교체하지 않는다.
[결과 보고서](../../reports/GEO_GROUP_CV_20261014.md), [기계 판독 요약](../../reports/geo_group_cv_summary.json), [사전 계획](PLAN.md), [감사 증거](../../reports/geo_group_cv_audit.json)를 먼저 읽는다.
기존 `docs/RESEARCH_STATUS.md`의 T3 진행중 문구는 오래된 스냅샷이다. 해당 문구로 완료된 T3/T5를 재실행하지 않는다.

| 검증 | C 평균 Macro F1 | G2 평균 Macro F1 | G2−C |
|---|---:|---:|---:|
| 기존 Stratified | 0.688687 | 0.690993 | +0.002306 |
| 2차 지역 복합 그룹 | 0.565146 | 0.568886 | +0.003739 |
| 1차 지역 그룹 | 0.365650 | 0.361539 | −0.004111 |

**새 상위 지역에서 기존 성능 유지 확인 실패.** 1차 지역 CV의 실제3등급 Recall은 C0.631%/G2 0.608%다. 3→1 감소만 보고 개선으로 판단하지 않는다. 하위 지역 CV의 G2 개선도 3/5Fold이며 원본995 집단은 악화됐다. 누수/암기의 인과적 입증이나 다른 지진 검증이 아니다.

## 보존된 조건

개발208,480행만 사용한다. `data/development.csv`, 기존 `data/split_manifest.csv`의 SHA를 설정 계약과 대조한다. 원래 홀드아웃 feature/label, 원본 전체 CSV는 필요 없으며 열지 않는다.
C/G2 원래 factory, LR 고정, 입력38개 및 age995 처리 그대로다. G2는 원본 연령 구간과 **geo1×기초구조**를 함께 추가한다.
모든 Fold에서 ID/그룹 중복0이며 SGKF1.8.0·seed42를 재구성해 manifest가 일치한다. 원핫 미관측 범주0 처리는 기존 정책이며 변경하지 않았다.

## 재학습 없이 기존 실행을 감사하기

제공된 비공개 패키지를 저장소 루트에 풀면 `outputs/geo_group_cv_v1/`와 `baseline_private/`가 생긴다. `baseline_private/`에는 기존 C/G2의 OOF·집계·메타데이터만 있으며 기존 모델은 없다. 개발 CSV와 기존 manifest는 원래 비공개 전처리 패키지에서 **그 두 파일만** `data/`에 준비한다. 홀드아웃을 같이 추출하지 않는다.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_geo_group_cv.py" -v
python -m experiments.geo_group_cv.audit --data data \
  --output outputs/geo_group_cv_v1 --baseline baseline_private \
  --destination outputs/geo_group_cv_reaudit
```

감사는 분류기를 fit하지 않는다. OOF ID/정답/Fold/확률, 별도 혼동행렬 산술, train-only 중앙값/스케일러/범주,20개 모델의 전체 검증행 재예측까지 검사한다. `joblib`은 신뢰할 수 있는 이번 생성 모델에만 사용한다. 출처 불명 모델을 로드하지 않는다.
이미 완료된 감사 destination은 덮어쓰지 않으므로 재감사마다 새 경로를 준다. 비공개 데이터가 들어 있는 체크아웃에서 기존 전체 integration suite를 무작정 실행하지 않는다. 기존 체크섬 테스트에는 이번에 접근 금지한 홀드아웃 경로가 포함돼 있다. 전체 합성 CI는 비공개 데이터 없는 원격 체크아웃에서 수행됐다.

## 20회 새로 재현할 때만

같은 실험을 중복 실행할 필요는 없다. 독립 로컬 재현을 명시적으로 할 때만 다음을 사용한다. 기존 산출물과 분리된 새 출력 경로를 쓴다.

```bash
python -m experiments.geo_group_cv.run prepare --data data --output outputs/geo_group_cv_repro
# 기존 기준선의 검증 기록을 재현 폴더에 복사한다. 원시 OOF의 SHA와 지표는 감사기가 다시 대조한다.
mkdir -p outputs/geo_group_cv_repro/evidence
cp outputs/geo_group_cv_v1/evidence/baseline_C_verified.json outputs/geo_group_cv_repro/evidence/
cp outputs/geo_group_cv_v1/evidence/baseline_G2_verified.json outputs/geo_group_cv_repro/evidence/
for scheme in geo2 geo1; do
  for variant in C G2; do
    python -m experiments.geo_group_cv.run fit --data data \
      --output outputs/geo_group_cv_repro --scheme "$scheme" --variant "$variant" \
      --code-commit 3ebe0f39f26d6151216ebd03d0d3f024a7bc29e5
  done
done
python -m experiments.geo_group_cv.audit --data data \
  --output outputs/geo_group_cv_repro --baseline baseline_private \
  --destination outputs/geo_group_cv_repro_audit
```

Windows PowerShell의 fit 반복문:

```powershell
foreach ($scheme in @('geo2','geo1')) {
  foreach ($variant in @('C','G2')) {
    python -m experiments.geo_group_cv.run fit --data data `
      --output outputs/geo_group_cv_repro --scheme $scheme --variant $variant `
      --code-commit 3ebe0f39f26d6151216ebd03d0d3f024a7bc29e5
    if ($LASTEXITCODE -ne 0) { throw '학습 실패: 같은 실험 폴더에서 자동 재시도하지 않음' }
  }
}
```

PowerShell에서 기준선 기록 복사는 `New-Item -ItemType Directory -Force outputs/geo_group_cv_repro/evidence`와 `Copy-Item outputs/geo_group_cv_v1/evidence/baseline_*_verified.json outputs/geo_group_cv_repro/evidence/`를 사용한다. 환경 패키지는 원격 requirements.txt에 고정돼 있다.
`--code-commit`은 실행기·설정·의존 원본 소스가 실제로 그 커밋과 같을 때만 위 값을 사용한다. 이후 실행기 변경은 기존 split seal에서 실패해야 정상이다. 결과/감사 문서만 추가한 후속 커밋은 학습 바이트를 바꾸지 않는다.

## 증거와 실행 횟수

- 학습 기준 코드 `3ebe0f39f26d6151216ebd03d0d3f024a7bc29e5`; 감사 코드 `83d5b98bd9196b00100683f99d25af24c1b1ea3e`.
- 신규 실데이터 학습20회, 기존 랜덤 CV 재학습0회, 학습 실패/재시도0회, 수렴20/20.
- T5 단위 테스트29개 통과. 코드737a17a 전체 CI264개 중235통과/29건너뜀/실패0. 별도 UI 워크플로도 성공했다.
- 첫 독립 감사 호출은 도구60초 제한으로 중단됐고 같은 코드로 다시 감사해 PASS했다. 모델 재학습은 없었다.
- private `design.json`과 `design_seal.json`, `geo{1,2}/manifest.csv`, `geo{1,2}/{C,G2}/`, `audit/`, `evidence/`에 원시 근거가 있다.
- 공개 요약·집계와 private 행별 OOF/모델을 구분한다. private ZIP이나 `baseline_private/`를 공개 Git에 추가하지 않는다.

## 다음 연구에서 하지 않을 일

이20회와 기존 C/G2 랜덤 Fold를 실적처럼 다시 학습하지 않는다. G2를 새 최종 모델로 자동 승격하지 않는다. 기존 홀드아웃을 열어 결론을 고치지 않는다. 지역 변수 제거/인코딩 대안/다른 모델의 Group CV는 별도 사전 계획과 연구 ID로 진행한다. 진행 중인 T4 결과를 확인 없이 가정하거나 수정하지 않는다.
