# T26 — CatBoost 범주형 직접 처리 비교

2026-10-15 사용자의 추가 실험 한 개 요청에 따라 선택했다. 개인 `peter`에서 구현·검토·커밋한 뒤 실자료 실행 전에 봉인한다. 봉인 뒤에는 변경하지 않고 진행 상태는 [RESEARCH_STATUS](RESEARCH_STATUS.md)에 기록한다.

## 질문과 비교 범위

T25는 동일 C 원핫 입력에서 CatBoost로만 바꾸었고 평균 Macro F1이 LGBM-O보다 낮았다. 이번에는 같은 CatBoost 설정을 유지하고 원핫 대신 원래 범주값을 직접 제공해 범주형 처리 방식의 효과를 비교한다. CatBoost 공식 문서는 외부 원핫 대신 범주형 기능 사용을 안내한다. 개선 여부는 이번 결과로 확인하며 공식 권고만으로 개선을 전제하지 않는다. [공식 범주형 안내](https://catboost.ai/docs/en/features/categorical-features).

신규 팔은 `CATBOOST_NATIVE_C` 하나다. 주비교는 T25 `CATBOOST_C`, 보조 비교는 기존 T8/T13 `LGBM_O`다. 주비교는 같은 모델의 표현 방식 비교이고, 보조 비교는 기존 기준 모델에 대한 결과다. 주비교만 개선했다고 기존 기준선도 개선했다고 말하지 않는다. 두 비교 모두 기존 OOF를 재사용하며 추가 기준선 학습은 없다.

## 고정 입력과 학습 경계

1. 원본 38개 입력을 보존하고 ID와 정답은 입력에서 제외한다. 기존 C의 age995 대치·표시, 수치 스케일링, 이진 처리를 유지한다. 통계는 각 outer-train에서만 학습한다.
2. C의 비범주형 28열(수치5, 이진22, age995 표시1)은 float32 값·순서가 기존 C와 정확히 같아야 한다. 원핫 전체 행렬을 밀집화하지 않으며 이 28열만 직접 생성한다.
3. 나머지11열은 기존 `CATEGORICAL` 순서(지역3·문자 범주8)로 정규화한 원래 문자열을 붙인다. 총39열 DataFrame이며 CatBoost에 범주형 열 위치28~38을 명시한다. 범주값을 수치 크기로 해석하거나 순서 인코딩하지 않는다.
4. 학습 어휘와 검증의 미관측 범주 건수·비율을 비공개 메타데이터에 기록한다. 미관측 값은 그대로 전달해 CatBoost가 처리하며 검증 자료로 어휘·빈도·정답 통계를 학습하지 않는다.
5. CatBoost 내부의 범주 통계·조합 처리는 고정 버전의 학습 알고리즘에 맡긴다. 원래 입력 외에 사람이 만든 신규 교차변수를 추가하지 않는다. 이 비교는 내부 범주 처리 전체의 효과이며 개별 통계나 조합의 효과를 분리하지 않는다.

CatBoost의 범주 통계는 학습 자료와 그 정답에서 학습되며, 일부 통계는 순열상 앞선 행을 이용한다. 이것은 예측할 행의 실제 피해 등급으로 입력 변수를 만드는 방식과 다르다. `fit`에는 outer-train과 해당 정답만 전달하고 `eval_set` 없이 학습한다. outer-valid는 학습 종료 뒤 예측에만 사용하며 validation 정답을 Pool·fit·전처리에 전달하지 않는다. [공식 범주 통계 계산 설명](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic), [Pool 입력 계약](https://catboost.ai/docs/en/concepts/python-reference_pool).

## 모델·분할·환경·예산

- 실험 ID `T26_catboost_native_c_v1`, 코드 `experiments/catboost_native_c/`, 출력 `local_reproduction/T26_20261015_catboost_native_c/run`. 출력은 최초 prepare 때 존재하지 않아야 한다.
- T25와 같은 CatBoost1.2.10 CPU, iterations300, learning_rate0.05, depth6, l2_leaf_reg3, random_strength1, Bayesian bootstrap, bagging_temperature1, Plain boosting, SymmetricTree, MultiClass, classes_count3, random_seed42, thread_count2, allow_writing_files=False, verbose=False를 유지한다. 생성 파라미터는 T25와 정확히 같고, 차이는 범주 입력과 `cat_features` 지정이다. 이에 따라 활성화되는 내부 범주 관련 기본값은 실제 get_all_params()로 기록하고 버전에 결속한다.
- 외부 sample/class weight·재표집·SMOTENC·튜닝·early stopping·사후 결정 규칙 없음. 내부012→실제123 확률 순서를 검증하고 argmax 동률은 낮은 등급으로 정한다.
- 동일 geo2 manifest seed42/101/202/303 × 5Fold = 신규20회 시도 상한, 자동·수동 재시도0회. 별도 실자료 smoke fit0회. 첫 공식 geo2_s42/f0을 자원·체크포인트 점검에 사용한다.
- T25 OOF는 실제 PASS_T25 감사, code_commit·design·analysis·ledger·OOF SHA와 ID·정답·Fold를 결속한 뒤 사용한다. LGBM-O도 기존 T8/T13의 SHA 고정 OOF만 재사용한다.
- 기존 frozen Python3.13.5와 T25 vendor를 그대로 재사용하며 패키지를 설치·업그레이드하지 않는다. 비공개 runtime_manifest_v1.json SHA `c67feaf2addb617dea50e2cf26a3757d0f1cab037add37611851893a5fbde6ef`로 버전·파일·실제 모듈 출처를 대조한다. 명세 ID가 T25인 것은 재사용 환경의 출처이며 실험 ID는 T26으로 분리한다.

## 사전 판정

주비교 T25 원핫과 보조 비교 LGBM-O에 아래 다섯 기준을 각각 적용해 별도 판정한다.

1. 20Fold 평균 Macro F1 차이≥+0.003.
2. 양의 차이 Fold≥14/20, 동률 제외.
3. 양의 seed별 Fold 평균 차이≥3/4.
4. 네 seed 혼동행렬을 합한 3등급 Recall 차이≥0.
5. 같은 반복 pooled의3→1 오류율 차이≤0.

주비교 모두 충족은 고정 범주 표현 개선 가설 지지, 보조 비교 모두 충족은 고정 기존 기준선 도전 조건 충족으로 기록한다. 어느 판정도 자동 채택이나 독립 최종 성능을 의미하지 않는다. 결과를 보고 어느 기준선을 주비교로 삼을지 바꾸지 않는다.

Fold 평균·SD(ddof=1), seed별 pooled, labels=[1,2,3]의 P/R/F1·혼동행렬, 비교별 오류 전환, age995/non995, 범주별 미관측 비율을 보고한다.3→1·3→2의 분모는 실제3등급,2→3은 실제2등급이다. 개발208,480행의4회 반복833,920예측은 독립 관측이 아니다. 같은 개발 분할을 반복 사용한 탐색이며 통계적 유의성·새 지역 전체의 일반화·안전성을 주장하지 않는다.

## 검토·실행·감사

- 합성 검사로 C 비범주28열 일치, 입력39열·범주11열 순서/자료형, train-only age/scale/어휘, 미관측 문자열, validation 정답 미전달, native300tree/classes/cat indices, 저장·재로딩 및 실패 원장을 확인한다. 작은 합성 학습과 실자료20회 예산은 구분한다.
- 비공개 입력 증거에 수치28열 바이트 SHA, 범주 문자열의 안정적 직렬화 SHA, 열 순서·자료형·전처리 상태·학습 어휘를 저장한다. 원시값·어휘·전체 상태는 공개하지 않는다.
- Fold 시작 전에 attempt1/running을 기록하고 native `.cbm`·전처리·예측·지표를 즉시 저장한다. SHA·백업·재로딩 확률 절대차≤1e-12를 확인한 뒤 complete로 기록한다. native 상태 float 직렬화는 T25와 동일하게 절대차≤1e-7 AND float32 값 정확 일치를 요구하고 나머지 자료형·키·구조는 엄격 비교한다.
- fits/finalizations 어느 곳이든 failed/running이면 다른 Fold 필터를 통한 신규 실행을 차단한다. 실패·부분 산출물을 보존하고 완료 Fold를 재학습하지 않는다.
- 전용 검사 → 독립 검토 → 전체 회귀 → 최종 소스·계획 커밋 → 봉인 → 첫 공식 Fold → 자원 재확인 → 나머지19 → finalize → 독립 감사 → 집계 보고·독립 검토 순서다.
- 독립 감사는 실행기 import·저장 joblib 역직렬화·새 학습 없이 수치/범주 입력을 재구축하고 native20모델을 재추론한다. 두 기준선·분할·환경·원장·원본/백업 SHA와 모든 집계·판정을 대조한다.

실자료 실행은 기존 Astra 실행 관리자 한 명만 담당한다. T25 종료·잠금 해제와 현재 자원을 먼저 확인한다. 새 Fold 전 가용RAM≥8GiB·디스크≥15GiB, 지속CPU≥90%·디스크 포화가 없을 때 실행하고 부족하면 후속 단위를 보류한다. 중복 관리자·중복 Fold를 실행하지 않는다.

홀드아웃 접근, 새 브랜치/worktree, 팀원 브랜치·메인 클론 checkout 변경, 기존 봉인 소스·결과 덮어쓰기, 원시자료·모델·OOF·manifest 공개 업로드는 금지한다. 결과는 독립 감사 후 `reports/CATBOOST_NATIVE_C_20261015.md`와 `catboost_native_c_20261015.json`에 집계만 남긴다. 공통 문서·커밋은 부모가 맡으며 공개 push와 컴퓨터 자동 종료는 수행하지 않는다.
