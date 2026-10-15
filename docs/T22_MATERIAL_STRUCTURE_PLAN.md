# T22 재료 조합과 구조별 층수 실험 계획

2026-10-15 사용자 승인. 개인 `peter`에서 구현·검토·커밋 후 첫 실자료 학습 전에 봉인한다. 실제 현황은 [RESEARCH_STATUS](RESEARCH_STATUS.md)에 기록하며 봉인 뒤 이 계획을 수정하지 않는다.

## 목적과 비교 범위

기존 무가중치 LGBM-O의 C 전처리 입력에 두 종류의 파생 블록을 각각 추가한다. T12의 재료 개수 요약과 달리 재료의 정체를 가진 쌍을 표현하고, T21의 범주 간 조합과 달리 범주별 층수 값을 표현한다. T21 결과와 무관하게 정의·판정을 고정하며 T21 피처, T18 튜닝, T20 가중치, T17 결정 규칙과 결합하지 않는다. 순서형 목표 학습은 이 실험에 포함하지 않는다.

## 고정 피처 정의

| 팔 | 정의 | 추가 블록 |
|---|---|---|
| `MATERIAL_PAIRS` | 아래 11개 재료 표시값의 모든 서로 다른 쌍의 곱 | 고정 55열, 원래 0/1 값 |
| `STRUCTURE_FLOORS` | `foundation_type` 및 `other_floor_type` 각각의 학습 어휘 원핫 × `count_floors_pre_eq` | 두 범주 어휘 크기의 합만큼 열, 원래 층수 값 |

재료는 `has_superstructure_` 접두어 아래 `adobe_mud`, `mud_mortar_stone`, `stone_flag`, `cement_mortar_stone`, `mud_mortar_brick`, `cement_mortar_brick`, `timber`, `bamboo`, `rc_non_engineered`, `rc_engineered`, `other` 순서다. 이 목록의 i<j 조합을 순서대로 사용한다. 정답이나 빈도로 쌍을 선택하지 않으며 상수 열도 계획에서 제거하지 않는다. 강도 순서나 공학적 취약도를 부여하지 않는다.

구조별 층수는 `foundation_type`, `other_floor_type` 순서로 각각 outer-train 범주 어휘를 학습한다. 범주 문자는 구조 강도 순서가 아니다. 검증의 미관측 범주는 해당 블록 전체 0으로 표현하고 발생 건수·비율을 기록한다. 원래 범주와 층수 입력은 C 블록에 남는다. 층수 0은 합성 경계값으로 허용하며 모든 곱이 0이다.

두 팔 모두 추가 블록을 스케일링하지 않고 float32 CSR로 변환해 기존 C 행렬 뒤에 붙인다. 전체 희소 행렬 밀집화 금지. 재료는 정확한 0/1, 층수는 유한한 비음수 정수, 범주는 결측 없는 문자열이어야 한다. 입력 스키마·열 순서와 비유한값을 검사한다. 학습용 y는 피처 계산에 사용하지 않는다. 기존 C의 995 대치와 표시 정책을 그대로 보존한다.

## 모델과 분할 및 예산

- 실험 ID `T22_material_structure_interactions_v1`, 코드 `experiments/material_structure_interactions/`.
- C 전처리 및 LGBM-O: 300 rounds, 31 leaves, learning_rate 0.05, 내부 2스레드. 나머지 설정과 동결 환경은 T8/T13 원핫 기준과 동일하다.
- 무가중치, 재표집 없음, early stopping·튜닝·임계값 조정 없음. labels=[1,2,3] 순서 argmax, 완전 동률은 낮은 등급.
- geo2 manifest seed 42/101/202/303 × 5 Fold × 두 팔 = 신규 **40회 시도 상한**, 팔마다 20개. 자동·수동 실패 재시도 금지. 완료 Fold는 검증 후 재사용한다.
- T8/T13 LGBM-O OOF를 SHA·ID·정답·분할 대조 후 재사용하며 기준선 신규 fit은 0회다.
- 출력 `local_reproduction/T22_20261015_material_structure/run`은 최초 봉인 때 미존재여야 한다. 기존 출력·태그를 덮어쓰지 않는다.
- 첫 `MATERIAL_PAIRS/geo2_s42/f0`을 공식 예산에 포함한 비용 점검으로 사용하며 별도 실자료 smoke fit은 없다.

## 사전 판정과 보고

각 팔을 동일 Fold의 무가중치 기준과 비교해 다음 다섯 조건을 모두 요구한다. T21 결과나 T20의 오류 1건 증가를 이유로 사후 기준을 바꾸지 않는다.

1. 20 Fold 평균 Macro F1 차이 ≥ +0.003.
2. 양의 차이 Fold ≥ 14/20(동률 제외).
3. 양의 seed별 Fold 평균 차이 ≥ 3/4.
4. 네 seed 혼동행렬을 합한 3등급 Recall 차이 ≥ 0.
5. 같은 반복 pooled의 3→1 오류율 차이 ≤ 0.

3→1·3→2의 분모는 실제 3등급 69,774 × 4 = 279,096, 2→3은 실제 2등급 118,607 × 4 = 474,428이다. 개발 건물 208,480개의 반복 예측 833,920개를 독립 관측으로 간주하지 않는다. Fold 평균·SD(ddof=1), seed별 pooled, 모든 등급 P/R/F1·혼동행렬, 오류 전환, age995/non995 결과를 남긴다. 다섯 기준 충족은 재사용 개발 자료의 가설 지지이며 최종 채택이나 안전성 보장이 아니다. 두 팔 중 좋은 결과만 보고하지 않는다.

## 검토와 실행 및 감사

- 합성 검사: 55쌍 순서·수기 곱, binary 검증, 0층/미관측 범주/빈 입력/열 순서, y 무사용, 저장·복원 및 CSR 상태를 확인한다.
- 원래 C 블록 불변을 확인하고 base/extra/full의 train/valid 여섯 CSR 증거, 피처 어휘·정의·상태를 저장한다. 각 Fold 모델·전처리·피처·예측·지표를 체크포인트로 저장하고 SHA·백업·재로딩 예측을 검증한다.
- 테스트 → 독립 검토 → 전체 회귀 검사 → 계획·코드 커밋 → 봉인 → 첫 공식 Fold → 자원 재확인 → 나머지 39개 → 집계 → 독립 감사 → 집계 보고 순서다.
- T21의 봉인 소스·설정·계획·감사 스크립트는 수정하지 않는다. 기존 의존 코드도 변경 없이 재사용한다.
- 실자료 실행은 기존 Astra 실행 관리자 한 명만 맡는다. T21 학습·무거운 감사와 공통 작업 잠금을 조율하고 중복 프로세스를 만들지 않는다. 가용 RAM≥8GiB·디스크≥15GiB, 지속 CPU≥90% 또는 디스크 포화가 없을 때만 새 Fold를 시작한다.
- 독립 감사자는 실행기 import 및 저장 joblib 역직렬화 없이 C·추가 피처를 재구성하고 40개 native 모델의 재추론, 여섯 CSR, 모든 집계·판정, 소스·입력·백업·시도 수를 대조한다. 감사 중 분류기 학습은 없다.
- 실패·봉인 불일치·예산 초과 시 후속 단계 중단, 로그·원본 보존. 완료 파일 삭제나 재학습으로 실패를 숨기지 않는다.

공통 현황·커밋은 컨트롤 타워가 맡는다. 공개 보고는 감사 PASS 이후 집계만 포함한다. 홀드아웃 접근, 새 브랜치·worktree, 팀원 브랜치 변경, 원시 데이터·모델·OOF·manifest 공개 업로드, 최종 모델 변경은 하지 않는다.

## 실행 관리자 명령

`P`는 workspace의 `local_reproduction/20261014_01/venv/Scripts/python.exe`, 작업 위치는 `worktrees/peter_rules`다. TEMP/TMP는 workspace의 비공개 폴더를 쓰고 UTF-8로 출력한다.

```text
P -m experiments.material_structure_interactions.run prepare --data ../../earthquake/data --output ../../local_reproduction/T22_20261015_material_structure/run --code-commit <검토된 전체 SHA>
P -m experiments.material_structure_interactions.run fit-all --data ../../earthquake/data --output ../../local_reproduction/T22_20261015_material_structure/run --arms MATERIAL_PAIRS --schemes geo2_s42 --folds 0
```

첫 Fold 저장·재로딩·백업·정상 종료와 자원 여유 확인 후 필터를 조절해 남은 Fold를 실행한다. 40개 완료·프로세스 종료·잠금 해제 뒤 같은 모듈의 `finalize`를 수행한다. 독립 감사 CLI는 컨트롤 타워가 별도 준비 영수증과 함께 인계한다. 이미 봉인한 출력에 `prepare`를 반복하지 않는다.
