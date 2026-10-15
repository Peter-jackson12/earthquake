# T25 — C 입력을 유지한 CatBoost 모델 계열 비교

2026-10-15 사용자 승인. 개인 `peter`에서 구현·검토·커밋한 뒤 실자료 실행 전에 봉인한다. 봉인 후 계획은 수정하지 않으며 현재 상태는 [RESEARCH_STATUS](RESEARCH_STATUS.md)에 기록한다.

## 목적과 비교 범위

기존 T8/T13 LGBM-O와 같은 개발 행·저장 분할·C 전처리 행렬을 사용하고 분류기만 CatBoost로 바꾼다. 팀원 노트북의 저장 점수는 평가 분할·전처리·파생변수 생성 근거가 달라 직접 비교하지 않는다. 팀원 브랜치와 파일은 변경하지 않는다.

정답으로 만든 파생변수, 신규 구조·지역 피처, SMOTENC, 클래스 가중치, 튜닝, early stopping, 사후 결정 규칙을 추가하지 않는다. 이 비교로 CatBoost 전체의 우열이나 같은 모델 용량에서의 우월성을 주장하지 않는다. 두 모델의 트리 구조·정규화·내부 학습 방식은 다르다.

## 고정 입력·분할

- 실험 ID `T25_catboost_c_v1`, 팔 `CATBOOST_C`, 코드 `experiments/catboost_c/`.
- 기존 C 전처리: outer-train에서 학습한 비995 연령 중앙값으로 995 대치, `age_is_995` 표시, 기존 수치 스케일링·원핫을 그대로 유지한다. `experiments.lgbm_native_cat.run.make_preprocessor('O')`와 동일한 CSR float32 행렬이며 추가 열은 없다. 전처리를 전체 개발 자료에 fit하지 않는다.
- 원본 38개 입력을 유지하며 ID와 정답은 모델 입력에서 제외한다. CatBoost에는 범주형 원본 열을 따로 지정하지 않고 기존 숫자 CSR를 전달한다. 전체 희소행렬을 밀집화하지 않는다.
- geo2 Group manifest seed 42/101/202/303 × 5 Fold = **신규 20회 시도 상한**, 자동·수동 재시도 0회. T8/T13 LGBM-O OOF를 SHA·ID·정답·분할 대조 후 재사용한다. 기준선 재학습과 별도 실자료 smoke fit은 0회다.
- 출력 `local_reproduction/T25_20261015_catboost_c/run`은 최초 봉인 시 미존재여야 한다. 첫 `geo2_s42/CATBOOST_C/f0`은 공식 예산에 포함하며 저장·재로딩·백업·정상 종료·자원 재확인 후 나머지 19개를 진행한다.

CatBoost 공식 문서는 자체 범주형 처리를 사용할 때 외부 원핫을 권장하지 않는다. 이번에는 모델 교체 효과를 분리하려고 기존 C 행렬을 의도적으로 유지한다. native categorical 비교는 이 실험에 포함하지 않는다. [공식 범주형 안내](https://catboost.ai/docs/en/features/categorical-features).

## 모델과 환경

CatBoost **1.2.10**, CPU, `iterations=300`, `learning_rate=0.05`, `depth=6`, `l2_leaf_reg=3`, `random_strength=1`, `bootstrap_type='Bayesian'`, `bagging_temperature=1`, `boosting_type='Plain'`, `grow_policy='SymmetricTree'`, `loss_function='MultiClass'`, `classes_count=3`, `random_seed=42`, `thread_count=2`, `task_type='CPU'`, `allow_writing_files=False`, `verbose=False`로 고정한다. 그 외 기본값은 버전으로 결속하고 실제 `get_all_params()`를 Fold별 비공개 지표에 기록한다. [공식 학습 파라미터](https://catboost.ai/docs/en/references/training-parameters/common), [배포 버전](https://pypi.org/project/catboost/1.2.10/).

Bayesian bootstrap은 고정한 CatBoost 학습 알고리즘의 내부 동작이다. 외부 데이터 증식·재표집·사용자 지정 sample/class weight는 적용하지 않는다. `eval_set` 없이 300회까지 학습하며 validation을 학습 중 선택·중단에 사용하지 않는다. 내부 정답은 0/1/2로 매핑하고 클래스 순서와 300 tree 완료를 확인한다. 보고 확률은 실제 등급 1/2/3 순서이며 argmax 동률은 낮은 등급이다.

기존 Python 3.13.5 동결 가상환경을 변경하지 않는다. 기존 NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, LightGBM 4.6.0, joblib 1.5.3, threadpoolctl 3.6.0을 유지한다. 새 패키지만 T25 전용 `vendor`에 설치하고 비공개 환경 manifest에 패키지 버전·원본 경로·파일 SHA를 기록한다. 기존 패키지를 vendor에서 대체하지 않는다. 환경 manifest SHA와 실제 모듈 경로·내용을 설계와 실행 검증에 결속한다. 환경 미확인 상태에서는 실자료 실행하지 않는다.

## 사전 판정과 해석

다음 다섯 조건을 모두 요구한다.

1. 20 Fold 평균 Macro F1 차이 ≥ +0.003.
2. 양의 차이 Fold ≥ 14/20, 동률 제외.
3. 양의 seed별 Fold 평균 차이 ≥ 3/4.
4. 네 seed 혼동행렬을 합한 3등급 Recall 차이 ≥ 0.
5. 같은 반복 pooled의 3→1 오류율 차이 ≤ 0.

labels=[1,2,3] Macro F1, Fold 평균·SD(ddof=1), seed별 pooled, 클래스별 P/R/F1·혼동행렬, 오류 전환과 age995/non995를 보고한다. 3→1·3→2의 분모는 실제 3등급, 2→3의 분모는 실제 2등급이다. 개발 208,480행을 네 번 평가한 833,920개 예측은 독립 표본이 아니다. 반복 사용한 개발 분할에서의 탐색이며 조건 충족도 독립 외부 성능·통계적 유의성·최종 모델 채택을 의미하지 않는다. 기존 연구 기준과 최종 모델을 자동 변경하지 않는다.

## 검토·보존·감사

- 합성 검사로 기존 C 행렬 동일성, 정답 미사용 전처리, 학습·검증 분리, 희소 입력, 클래스 순서, 고정 파라미터·반복 수, native 저장·재로딩, 실패 원장과 중복 실행 방지를 확인한다. 합성 소형 학습은 연구 20회 예산과 구분하며 실자료 smoke fit은 금지한다.
- 각 Fold 학습 전에 running/attempt1을 기록한다. native `.cbm`, 전처리, 행별 예측·지표·로그를 즉시 저장하고 SHA-256·재로딩 최대 확률 절대차≤1e-12·원본/백업 일치를 확인한 뒤 complete로 기록한다.
- train/valid C CSR 배열의 shape/dtype/SHA와 실제 모델 파라미터·클래스·tree 수를 남긴다. 어느 학습·집계든 failed/running이면 연구 전체 신규 작업을 차단한다. 완료 Fold는 재학습하지 않고 실패 기록·부분 산출물을 보존한다.
- 합성 검토에서 native 저장 시 설정값의 십진 반올림을 확인했다. 설정 상태의 float끼리만 절대차≤1e-7이면서 float32 변환값이 정확히 같은 경우를 허용하고, 키·타입·문자열·정수·구조는 일치를 요구한다. 이는 모델 설정 저장 표현의 비교이며 예측 확률의 1e-12 기준은 유지한다. 합성 초기 비교 실패와 최종 성공은 별도 비공개 영수증에 보존한다.
- 전용 검사 → 독립 검토 → 전체 회귀 → 코드·계획 커밋 → 봉인 → 첫 공식 Fold → 자원 재확인 → 나머지 → finalize → 독립 감사 → 집계 보고·독립 검토 순서다.
- 독립 감사는 실행기 import·저장 joblib 역직렬화·새 학습 없이 C를 재구축하고 native CatBoost 20모델 재추론으로 검증한다. 환경·분할·참조·원장·원본/백업 SHA, CSR, 확률·지표·다섯 판정을 대조한다.

실자료 prepare/학습/감사는 기존 Astra 실행 관리자 한 명만 담당한다. T24 완료·프로세스·잠금 상태와 조율하고 중복 실행을 금지한다. 새 Fold 전 가용 RAM≥8GiB·디스크≥15GiB, 지속 CPU≥90% 또는 디스크 포화가 없는지 확인한다. 자원이 부족하면 후속 단위를 보류한다.

홀드아웃 접근, 기존 봉인 소스·출력 변경, 새 브랜치/worktree, 팀원 브랜치 변경, 원시자료·모델·OOF·manifest 공개 업로드는 금지한다. 감사 PASS 후 `reports/CATBOOST_C_20261015.md`와 `catboost_c_20261015.json`에 집계만 작성한다. 공통 문서·커밋은 컨트롤 타워가 담당하며 공개 push는 이번 범위에 없다.
