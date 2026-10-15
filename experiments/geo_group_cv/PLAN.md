# T5 지역 Group CV v1 — 학습 전 고정 계획

2026-10-14 KST. 기준 T3: `a4fc5c82a1907d8fc975f0971deaed8d26a4461f`.
독립 브랜치: `peter-exp/geo-group-cv-v1`. main/peter/T3/T4는 수정하지 않는다.

## 질문과 비교

동일한 개발 208,480행에서 지역 분리 시 기존 LR 정책 C와 G2의 성능이 유지되는가?
1차 실행은 `(geo_level_1_id, geo_level_2_id)` 복합 그룹, 2차 실행은 `geo_level_1_id` 그룹이다.
두 방법 모두 sklearn 1.8.0 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`를 사용한다.
입력 행은 building_id 오름차순으로 고정하고, 그룹은 정수 지역 튜플을 정렬한 순서대로 정수 코딩한다.
ID는 정렬·검증에만 쓰고 모델 입력에서는 제외한다. 그룹 코딩은 정답을 사용하지 않는다.

## 모델 결과 확인 전 분할 판정

분할 전에 그룹 수·크기·지역 계층 관계를 조사한다. 다음 기준은 모델 학습/예측 전에 고정한다.
- 즉시 실패: 그룹 5개 미만, 빈 Fold, 그룹/ID 학습-검증 중복, ID 누락/중복, 훈련 또는 검증에 등급 1/2/3 누락, 각 행 OOF 배정 1회 위반.
- 실행 중단: 어느 훈련/검증 등급 support가 30 미만이거나, 검증 크기가 개발의 1% 미만 또는 50% 초과.
- 불균형 경고: 검증 크기 최대/최소 비가 3 초과 또는 전체 개발 대비 Fold 등급 비율 절대 차이가 0.10 초과.
- 경고만 발생하면 seed를 바꾸지 않고 분포 이동 스트레스 테스트로 한계를 명시한 뒤 실행한다. 중단 기준이면 해당 분할은 미실행으로 남긴다. 유리한 대체 분할/seed 사후 탐색 금지.
위 수치는 연구 실행을 위한 사전 운영 기준이지 보편적 통계 정리가 아니다.
두 분할의 구성·support·편차·판정과 SHA를 모두 저장한 뒤 첫 분류기를 학습한다.

## 보존과 재사용

개발 CSV SHA256 `8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1`.
기존 manifest SHA256 `7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721`.
원래 52,121행 홀드아웃의 feature/label은 추출·읽기·재평가하지 않는다. 원본 전체 CSV도 필요하지 않다.
load_development()로 개발 자료만 읽고 원래 manifest는 변경하지 않는다.
C와 G2의 기존 Stratified OOF를 SHA·ID·정답·Fold·확률·설정 검증 후 기준으로 재사용한다. 기존 분류기 재학습은 0회다.

## 고정 학습

C: experiments.age995.run.make_pipeline('C').
G2: experiments.age995.interactions.run.make_pipeline('G2').
G2는 원본 연령 구간과 1차 지역×기초구조를 함께 추가하므로 C–G2 차이를 교차항 하나의 순수 효과로 해석하지 않는다.
LR C=1, lbfgs, max_iter=1500, tol=1e-4, random_state=42. 38개 입력 보존, 가중치·샘플링·튜닝·임계값 변경 없음.
모든 중앙값/스케일러/사전 fit은 훈련 Fold 한정. 희소 CSR 유지, 학습 단일 스레드.
2개 분할 × 2개 설정 × 5Fold = 최대 신규 20회 분류기 학습. 실패 시 같은 실험 ID에서 몰래 재시도하지 않는다.

## 평가와 독립 감사

Fold Macro F1 평균/표준편차(ddof=1), pooled OOF F1/Accuracy, 등급별 P/R/F1/support, 원시/행정규화 혼동행렬.
3→1/3→2의 분모는 실제3등급이다. 원본995/비995, 모든1차 지역 및 사후 진단지역26/20/13을 기록한다.
미관측 지역/범주/추가 조합 수와 각 원핫 블록의 0 처리, 훈련 통계/사전 불변, 모델 재로딩 후 전체 검증Fold 예측·확률 일치를 확인한다.
원시 OOF에서 별도 산술로 모든 지표를 재계산한다. C/G2의 직접 비교는 반드시 동일 Group CV 안에서 한다.
Group CV 점수 하락만으로 기존 누수·지역암기를 인과적으로 확정하지 않는다. 건물 수 가중 pooled와 동일 가중 Fold 평균을 구분한다.
통계적 유의성/새 독립 최종성능/타 지진 일반화/운영 안전성을 주장하지 않으며 최종 모델을 자동 교체하지 않는다.

## 산출물과 출처

experiments/geo_group_cv/{config.json,run.py,audit.py,README.md}, tests/test_geo_group_cv.py, reports/GEO_GROUP_CV_20261014.md 및 집계 증거.
Fold manifest·행별 OOF·모델·전체 로그는 공개 Git에 올리지 않고 비공개 패키지로 분리한다.
공식 분할 정의: https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
기존 근거: reports/AGE995_INTERACTIONS_20261014.md, reports/AGE995_DIAGNOSTICS_20261014.md.
RESEARCH_STATUS의 T3 진행중 문구는 오래된 스냅샷이며 실제 완료 보고서가 우선한다.
