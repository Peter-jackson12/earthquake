# T4 모델 계열 A/C 비교 v1 — 실학습 전 고정 계획

2026-10-14 KST. Base `a4fc5c82a1907d8fc975f0971deaed8d26a4461f`.
전용 브랜치 `peter-exp/model-family-comparison-v1`. 기존 LR G1~G5(T3) 파일·결과와 분리한다.
Base의 최신 커밋에는 T3 결과가 있으나 RESEARCH_STATUS의 T3 설명은 이전 스냅샷이다. 이 연구의 비교 후보에 G1~G5를 넣거나 그 파일을 수정하지 않는다.

## 사전 설계

- 신규 `XGB_A_v1`, `XGB_C_v1`, `LGBM_A_v1`, `LGBM_C_v1`: 4설정×고정5Fold=20회 fit.
- `A`: 원본 age 유지. `C`: 각 학습 Fold의 비995 중앙값(0 포함)으로 995만 대체+원본995 표시. 기존 `make_pipeline(mode)['preprocess']`를 그대로 사용한다.
- 기존 LR A/C는 저장 OOF SHA·ID·정답·Fold·확률·메타데이터·환경·모델 파라미터를 재검증한 뒤 재사용한다. LR 신규 학습 0회.
- 개발208,480행, 학습166,784/검증41,696행, seed42, 기존 split manifest 유지. 데이터 계약 SHA를 먼저 검사한다. 원본·개발 CSV와 기존 결과는 덮어쓰지 않는다.
- 홀드아웃 feature/label은 압축 해제·읽기·재평가하지 않는다. Split manifest의 원래 홀드아웃 ID/-1 표시는 분할 무결성 검사에만 사용한다.
- 두 모델 모두 300 boosting rounds, learning_rate=0.05, seed42, CPU2 threads, outer 순차 실행. XGB hist/max_depth6; LGBM num_leaves31/max_depth=-1. 전체 파라미터·실행 버전은 config.json에서 고정한다.
- 이것은 미튜닝 고정 예산 설정이며 라이브러리 무인자 기본값이나 두 모델의 동일 표현력/동일 계산량을 뜻하지 않는다. 검증 점수를 보고 설정을 바꾸지 않는다.
- 무가중치·무리샘플링·무early-stopping. 외부 검증 Fold를 학습 함수에 eval_set으로 넘기지 않는다. Optuna·임계값 튜닝·교차 특성·E2 추가는 v1 범위 밖이다.
- 같은 mode/Fold의 전처리를 한 번만 fit하여 XGB와 LGBM에 같은 CSR float32 행렬을 전달한다. 수치5개 표준화, 범주/지역코드 Fold내 OHE, 원본38개 입력을 유지한다. 행렬 SHA와 학습 사전·중앙값·스케일러 통계를 기록한다.
- XGB의 희소 미저장 원소는 tree booster에서 missing으로 취급되고 LGBM은 zero_as_missing=false이다. 공통 CSR을 쓰되 이 라이브러리 의미 차이를 기록한다. 결과는 해당 구현·설정의 비교이지 순수한 트리 성장 방식만의 인과 효과가 아니다.
- 내부 라벨0/1/2를 사용하고 외부 예측/확률 순서는1/2/3으로 복구한다.

## 지표·결정

주지표: Fold Macro F1 평균/표준편차(ddof=1). Pooled OOF는 별도 계산한다.
전체/원본995/비995의 등급별 P/R/F1·support·원시/행정규화 CM, 3등급Recall,
3→1/3→2 건수·실제3등급 분모, 예측등급 분포를 기록한다. 지역1·기초구조·원본연령구간의 표본 수/오류/악화도 보존한다.

사전 비교: 각 가족 C−A, 각 신규 모델−같은 정책 LR, XGB−LGBM 같은 정책. Fold별 차이와 pooled 예측 전이(기존3→1 수정/신규/잔존)를 구분한다.
평균이 높은 후보를 수치상 후보로 표시하되 전체3→1·3→2·3등급Recall·995 집단의 부작용을 함께 제시한다. 작은 차이를 통계적 유의성/다른 지진 일반화/운영 안전성으로 주장하지 않는다. 팀·개인 최종 모델·앱·연구 C 기준을 자동 교체하지 않는다.

## 감사·보존

실행기는 sklearn 지표를 사용한다. 독립 감사기는 pandas crosstab과 직접 수식으로 재계산하고,
OOF SHA/ID/Fold/정답/확률/분모, 모델 파일 SHA와 전체 검증Fold 재추론 일치를 확인한다.
원시 OOF·저장 모델·전처리기는 비공개 출력에 보관한다. 합성 계약/손상 거부 테스트는 실성능 결과와 분리한다.
학습전후 원본 개발 파일·기존 LR·상속 소스 SHA를 비교한다. 코드·설정·집계·감사 요약만 공개 Git에 반영한다.
Git 작업은 대상브랜치 expected HEAD, non-force, base tree 보존. main/peter/기존age995 브랜치에 쓰지 않는다.

## 공식 기술 근거

- https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html
- https://xgboost.readthedocs.io/en/release_3.1.0/faq.html
- https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html
