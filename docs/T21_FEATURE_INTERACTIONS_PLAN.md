# T21 구조와 연령 상호작용 피처 실험 계획

2026-10-15 작성. 사용자가 두 후보의 구현·독립 검토와 자원 여유 시 실행을 승인했다. 개인 작업은 `peter`에서 진행한다. 아래 설계와 소스를 검토·커밋한 뒤 첫 실자료 학습 전에 봉인한다. 실제 진행 상태는 [연구 현황](RESEARCH_STATUS.md)에 기록하며 봉인 후 이 문서를 수정하지 않는다.

## 목적과 기존 실험

고정 LGBM-O에 새로운 상호작용 표현만 추가했을 때 geo2 Group CV가 개선되는지 비교한다. T18 튜닝·T20 가중치·T17 결정 규칙과 결합하지 않아 피처 추가 효과를 구분한다.

- [T3](../reports/AGE995_INTERACTIONS_20261014.md)는 연령 구간, geo1×기초구조, geo1×연령구간, 기초구조×연령구간을 비교했다. G2는 C 대비 F1 +0.002306이었으나 3→1 증가와 age995 하위 집단 악화로 보수적 선택 기준을 통과하지 못했다. G2−G1은 +0.000917로, G2−C 전부를 기초구조 교차항 효과로 해석하지 않는다.
- [T12](../reports/STRUCT_FEATURES_20261014.md)는 재료·용도 개수와 높이/층수, 면적×층수, 가구수/층수를 LR/LGBM에서 90회 학습했다. 공동 개선 후보는 0/6, 최고 개선은 stratified 약 +0.0009, geo2 약 +0.0011이었다. 이는 해당 묶음에 한정한 결과이며 피처엔지니어링 전체의 실패를 뜻하지 않는다.
- [기존 인벤토리](FEATURE_ENGINEERING_INVENTORY_20261014.md)는 T1~T6 당시 기록이다. T12까지 포함한 전체 목록으로 읽지 않는다.

## 벤치마크에서 확인한 것

[Sahil Ali 공개 앱 코드](https://github.com/sahilali09/earthquake-damage-prediction/blob/1b7cdb4b4bc733182cb9aae9f85ed7a64ad7cba0/app.py#L41)는 2026-10-15 현재 커밋 `1b7cdb4b4bc733182cb9aae9f85ed7a64ad7cba0`에서 아래 정의를 사용한다. 공개 코드를 읽었으며 외부 프로그램을 실행하거나 모델을 내려받지는 않았다.

| 외부 정의 | T12와 비교 | 이번 판단 |
|---|---|---|
| 약한 재료 4개 합: adobe_mud, mud_mortar_stone, mud_mortar_brick, bamboo | T12의 6개 목록과 다름 | 정확한 외부 묶음은 미실험. 우선 보류 |
| 강한 재료 3개 합: cement_mortar_brick, cement_mortar_stone, rc_engineered | T12의 4개 목록과 다름 | 정확한 외부 묶음은 미실험. 우선 보류 |
| material_quality_score = 강한 재료 합 − 약한 재료 합 | T12에 이 열은 없음. 원본 플래그의 선형 조합 | 검증된 재료 강도로 해석하지 않음 |
| total_secondary_uses = 종합 플래그와 세부 10개 합 | T12는 종합 플래그 제외 | 용도 수와 종합 여부가 섞여 우선 보류 |
| vulnerability_index = 정리한 age × height_percentage × count_floors_pre_eq | T3/T12에 없는 곱 상호작용 | A에서 C 연령 정책에 맞춰 적용 |

외부 앱은 995를 15로 고정 대체한다. A는 우리 C 정책의 학습 Fold 중앙값을 사용하므로 외부 프로젝트의 정확한 재현이 아니다. 외부 최종 모델 점수는 각 파생변수의 분리 효과를 입증하지 않는다.

[다른 LGBM 벤치마크](https://github.com/kuchbhi-xyz-hub/Earthquake-Damage-Prediction)의 README는 범주 처리·튜닝·결정 규칙을 주로 설명한다. 그 문서만으로 미검토 파생변수가 전혀 없다고 판단하지 않는다.

[공식 스키마](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/)에서 높이·면적은 정규화된 값이며 범주 코드는 익명화돼 있다. 높이 코드를 미터로 읽거나 기초구조 문자에 공학적 강도 순서를 부여하지 않는다.

## 비교할 두 묶음

| 팔 | 추가 피처 | 검증할 가설 |
|---|---|---|
| A `AGE_HEIGHT_FLOORS` | `age_clean * height_percentage * count_floors_pre_eq` 한 열 | 세 변수의 곱 표현이 고정 트리의 분할을 도울 수 있는가 |
| B `STRUCTURE_PAIRS` | `(foundation_type, roof_type)`와 `(ground_floor_type, other_floor_type)` 두 복합범주의 원핫 열 | 구조 조합을 명시하면 기존 개별 범주보다 유용한 분할을 만들 수 있는가 |

A의 `age_clean`은 기존 C 전처리와 동일하다. 원시 age=995만 outer-train의 non995 중앙값(0 포함)으로 대체하고 원래 `age_is_995`를 유지한다. 중앙값을 전체 개발 자료에서 계산하거나 15로 고정하지 않는다. 곱은 원래 단위의 age와 정규화 높이 코드·층수로 계산한 뒤 train-only StandardScaler를 적용한다. 이 열은 검증된 취약도 지수가 아니므로 `age_height_floors_product`로 명명한다.

B의 조합은 문자열 충돌이 없는 튜플 기반으로 정의한다. 어휘는 outer-train에서만 학습하며 검증의 미관측 조합은 모두 0으로 변환하고 건수·비율을 기록한다. 희귀 조합을 정답에 따라 고르거나 전체 개발 자료로 어휘를 만들지 않는다. 두 팔 모두 기존 38개 입력과 C의 기존 열을 그대로 보존하고 새 열만 뒤에 붙인다. 전체 희소 행렬을 밀집 배열로 바꾸지 않는다.

재료 11종의 동시출현 55개 항과 범주별 층수는 후속 보류 후보다. 이번 A/B 결과를 보고 같은 연구 안에 변수를 추가하거나 A+B 결합을 선택하지 않는다.

## 고정 모델과 학습 예산

- LGBM-O 기준: C 전처리, 300 rounds, 31 leaves, learning_rate=0.05, 내부 2스레드. 나머지 설정도 T8/T13 기준과 동일하게 복사·검증한다.
- 가중치 없음, ROS 없음, early stopping·튜닝·결정 규칙 변경 없음. labels=[1,2,3] 순서 argmax이며 동률은 낮은 등급이다.
- 동일 geo2 manifest seed 42/101/202/303 × 5 Fold × 두 팔 = **신규 40 fit 상한**. 완료 Fold는 재사용하고 자동 재시도는 없다. 실패는 기록을 보존하고 중단한다.
- 기준선은 T8/T13의 기존 LGBM-O OOF를 SHA·ID·정답·분할 대조 후 재사용한다. 새 기준선 학습은 0회다.
- 출력 예정: `local_reproduction/T21_20261015_feature_interactions/run`. 실행 직전 미존재를 확인하며 기존 경로를 덮어쓰지 않는다.
- 별도 실자료 smoke fit을 숨겨서 추가하지 않는다. 합성 검사 후 최초 정식 Fold를 비용 점검으로 사용하며 40회 안에 포함한다.

## 고정 판정과 결과 기록

각 팔을 자기 무가중치 기준선과 독립적으로 비교한다. T20과 같은 기술 기준 다섯 가지를 첫 실행 전에 코드·설정과 함께 동결한다.

1. 20 matched Fold 평균 Macro F1 차이 ≥ +0.003.
2. 양의 차이 Fold ≥ 14/20. 동률은 승리로 세지 않는다.
3. 양의 seed별 Fold 평균 차이 ≥ 3/4.
4. 반복 pooled 3등급 Recall 차이 ≥ 0.
5. 반복 pooled 3→1 오류율 차이 ≤ 0.

반복 pooled는 4개 seed 혼동행렬의 합에서 계산한다. 3등급 Recall과 3→1·3→2 오류율 분모는 실제 3등급 `69,774 × 4 = 279,096`, 2→3 오류율 분모는 실제 2등급 `118,607 × 4 = 474,428`이다. 실제 지지 수를 입력 계약과 대조해 확인한다. 이 기준이 모든 seed나 age995 하위 집단의 개선을 요구하거나 보장하는 것은 아니며 해당 결과도 별도로 공개한다.

모두 충족해도 고정 가설의 개발자료상 지지이며 최종 채택이나 안전성 인증이 아니다. T20의 오류 1건 증가 결과 때문에 마지막 기준을 완화하지 않는다. 미충족 여부와 실제 변화 크기를 함께 보고한다.

Fold 평균·SD(ddof=1), seed별 pooled, 각 클래스 P/R/F1·혼동행렬, 3→1·3→2·2→3의 실제등급별 분모, age995/non995 하위 집단, 기준 대비 오류 전환을 남긴다. 동일 건물 208,480개를 네 번 재분할한 값이므로 반복 예측 833,920개를 독립 표본으로 세지 않는다. 두 후보 비교와 누적 연구의 선택 편향을 명시한다. 개선 원인을 피처 중요도만으로 확정하지 않는다.

## 실행 준비 완료 조건

- A: 995 중앙값이 학습 행에서만 계산되는지, 비995·0 연령·995 행의 손계산, 기존 C 블록 불변, finite 결과 검사.
- B: 조합 충돌·미관측 범주·행 순서·훈련 어휘·희소 표현·기존 블록 불변 검사.
- 독립 검토와 전체 회귀 검사 후 학습 전에 코드·설정·계획 커밋, 입력·분할·기준선·소스 SHA 봉인.
- 각 Fold 저장·재로딩 확률 일치·원본/백업 SHA 확인, 실행 원장과 실험 잠금.
- 실행기 import 없이 변수식·train-only 통계·행렬·모델 재추론·집계를 대조하는 독립 감사 준비.
- 실자료 학습은 기존 실행 관리자 한 명에게 배정한다. T16/T18 감사와 자원을 조율하고 같은 실험을 두 주체가 실행하지 않는다. 검토·회귀 검사 PASS와 커밋 SHA를 인계받은 뒤 prepare를 수행한다. 가용 RAM≥8GiB, 디스크≥15GiB, 지속 CPU≥90% 또는 디스크 포화 없음과 공통 감사 잠금을 확인한다. 첫 정식 Fold의 저장·재로딩·백업 및 종료 후 자원을 확인하고 나머지를 진행한다. 자원 압박 시 새 Fold 시작을 보류한다.

홀드아웃 접근·팀 브랜치 변경·원시 자료/모델/OOF/manifest 공개 업로드·최종 모델 변경은 범위 밖이다.

## 실행 관리자용 순서

실행 모듈은 `experiments.feature_interactions.run`, 추가 피처는 실행기와 분리된 `experiments.feature_interactions.features`다. `P`는 기존 동결 Python이며 작업 위치는 `worktrees/peter_rules`다. TEMP/TMP를 workspace의 비공개 폴더로 지정하고 UTF-8 출력을 사용한다.

```text
P -m experiments.feature_interactions.run prepare --data ../../earthquake/data --output ../../local_reproduction/T21_20261015_feature_interactions/run --code-commit <검토된 전체 SHA>
P -m experiments.feature_interactions.run fit-all --data ../../earthquake/data --output ../../local_reproduction/T21_20261015_feature_interactions/run --arms AGE_HEIGHT_FLOORS --schemes geo2_s42 --folds 0
```

첫 Fold 성공과 자원 여유를 확인한 뒤 `--arms`, `--schemes`, `--folds` 필터를 조절하거나 제거해 남은 39개를 진행한다. 완료 Fold는 SHA 확인 후 건너뛴다. 40개 완료·프로세스 종료·잠금 해제 후 같은 모듈의 `finalize`를 수행한다. 독립 감사는 관리자에게 인계한 별도 감사 스크립트로 진행하며, 그 PASS와 집계 보고 검토 전에는 결과를 확정하지 않는다. 이미 존재하는 출력에 `prepare`를 다시 실행하지 않는다.
