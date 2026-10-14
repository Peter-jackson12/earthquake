# T6 지역 변수 Ablation v1 — 학습 전 고정 계획

2026-10-14 KST. 기준 T5 `acd5ecdeefeff619c2fae9639f1627638de59383`; 독립 브랜치 `peter-exp/geo-ablation-v1`.

## 질문과 범위

지역 변수의 단계적 제거가 미관측 지역 일반화와 실제 3등급 식별을 개선하는가? C는 기존 세 지역 변수를 모두 사용하는 LR이며 G2가 아니다. R12는 geo3, R1은 geo2/geo3, R0은 geo1/geo2/geo3을 모델 변환 대상에서 제외한다. CSV와 입력 스키마 38개는 유지한다. 원본 C factory에서 새 파이프라인을 생성한 뒤 categorical selector만 축소하고 remainder=drop을 검증한다. 지역 파생항은 추가하지 않는다. building_id/정답/Fold/group 키는 예측 변수에 허용하지 않는다.

## 보존 조건

기존 C의 fold-local 비995 연령 중앙값 대체+age_is_995, 수치형 스케일링, 범주 원핫(handle_unknown=ignore), float32 CSR를 유지한다. LR C=1, solver=lbfgs, max_iter=1500, tol=1e-4, random_state=42, class_weight=None. 튜닝·가중치·샘플링·임계값 변경 없음. 원본 preprocess.py v1.0.1과 T1~T5 코드는 수정하지 않는다. T4는 조회만 하며 모델·결과를 통합하지 않는다. main/peter/T3/T4/T5를 수정하지 않는다.

## 고정 데이터와 분할

개발 208,480행만 사용하며 building_id 숫자 오름차순으로 정렬한다. 기존 홀드아웃 52,121행의 feature/label과 원본 전체 CSV는 읽거나 추출하지 않는다.

- development.csv SHA256: 8ef0302a90fc66a49f20fdaefc8e7302ef4bbd3ca2b8b7b741a52f0503ce31b1
- Stratified manifest SHA256: 7673dbeb136e9a916c67a3b19928b3fbc621ae627a5667a677a1ca6eae838721
- geo2 복합 그룹 (geo1,geo2), 1,408개, T5 manifest SHA256: 36a40ddf1153b7a25bf36d5573b26b4dc9a0d67d240869c2e5cc1d85c1f89d64
- geo1 그룹 31개, T5 manifest SHA256: 1b4616ff7d5e052ac76707e58de48e3d0b23fb7a49e350c14d6704cc1ee488f5

T5 manifest의 원본 바이트와 전 행 ID→Fold 배정을 재사용한다. 재구성 검증에는 sklearn 1.8.0 StratifiedGroupKFold(5, shuffle=True, random_state=42), 숫자 지역 튜플 사전순 코딩을 사용한다. 유리한 seed/분할 탐색은 금지한다. 동일 검증 방식의 모든 모델은 같은 Fold를 사용한다.

## 학습 예산 및 사전 봉인

기존 C 15회 결과는 OOF SHA·ID·정답·Fold·확률·모델 설정·원본 코드와 지표를 확인한 후 재사용한다. C/G2 분류기 재학습 0회. 신규 R12/R1/R0 × stratified/geo2/geo1 × 5Fold = 최대45회. 각 fit 단일 스레드, 순차 실행. 계획/설정/실행 코드/의존 코드/환경/기준선/분할을 design.json과 SHA seal로 학습 전에 고정한다. 실제 학습 전후 시각과 시도·완료·실패 횟수를 남긴다. 기존 출력은 덮어쓰지 않는다.

## 평가와 판정

주지표는 labels=[1,2,3]의 Fold Macro F1 산술평균, 표준편차 ddof=1. pooled OOF F1/Accuracy, 등급별 P/R/F1/support, 원시/행정규화 혼동행렬을 별도로 보고한다. 실제3등급 Recall, 3→1/3→2 건수와 실제3등급 support를 분모로 한 비율을 함께 비교한다. 원본995/비995, 31개 전체1차 지역 및 26/20/13, Fold별 지표와 C 대비 차이, 같은 행의 오류 수정/신규 발생 및 예측 전이를 기록한다. 평균+보조지표+Fold 일관성+지역별 손익을 함께 읽고 임의 비용가중치·허용 하락폭·통계적 유의성을 사후 도입하지 않는다. 최종 모델 자동 교체·배포 없음.

## 검사와 중단 기준

데이터/소스/분할/기준선 SHA 불일치, ID/정답/Fold/확률 부정합, 그룹 또는 ID 누수, 미관측 범주의 비정상 처리, 제외 피처 재유입, train-only 통계 위반, 저장 모델 재예측 불일치가 발생하면 중단한다. T5 분할 운영 기준을 유지하며 빈 Fold/등급누락/support<30/검증 비율 1~50% 범위 위반도 중단한다. 비수렴 또는 fit 예외를 기록하고 자동 재시도·solver/iteration 변경을 하지 않는다. 합성 테스트와 실제 개발 학습을 분리한다.

독립 감사는 학습용 sklearn metrics를 재사용하지 않는 혼동행렬 산술, 전체 OOF 정합성, 45개 신규 모델 전체 검증행 재로딩, 원본 C 전처리에서 제외 열만 제거한 행렬과의 정확한 일치, 제외 지역 값 변경 시 행렬·예측 불변을 확인한다. C 기준선의 저장 모델이 있는10개는 재로딩하며, Stratified C의 과거 모델5개가 없는 경우 이를 명시하고 해당 모델을 새로 fit하지 않는다. 공통 행렬 검증을 위한 전처리-only fit은 분류기 학습과 구분한다.

## 산출물과 한계

공개 Git에는 신규 코드/테스트/계획/집계 보고서만 추가한다. 행별 OOF·manifest·joblib·원시 로그는 비공개 ZIP으로 분리한다. README가 T6의 재개 진입점이며 기존 연구 상태 문서의 오래된 진행중 표시는 완료 연구 재학습 근거로 사용하지 않는다.

지역 제거 개선은 누수/과적합의 인과적 증명이 아니다. 훈련 구성·지역 피해 특성·미관측 범주·정규화된 계수의 공동 재학습이 함께 작용할 수 있다. 이 연구는 기존 개발 데이터 안의 탐색이며 새 지진/외부 지역/독립 홀드아웃/운영 안전성 검증이 아니다.
