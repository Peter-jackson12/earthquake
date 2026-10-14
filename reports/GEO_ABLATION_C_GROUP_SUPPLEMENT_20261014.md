# T6S 지역 변수 Ablation — C Group CV 행 단위 보완 감사

2026-10-14 KST · `Peter-jackson12/earthquake` · 브랜치 `peter-exp/geo-ablation-c-group-supplement-v1` · 실험 ID `T6S_geo_ablation_c_group_supplement_v1`

## 1. 판정

**PASS.** 분류기 학습 0회, 홀드아웃 접근 없음.

[T6R 보고서](GEO_ABLATION_LOCAL_REPRO_20261014.md) §7의 한계 1을 보완한다. 해당 한계는 "C geo2/geo1 행 단위 OOF와 C 모델 부재로 Group CV의 같은 행 비교를 하지 못했다"는 것이다. 회수한 T5 비공개 패키지로 C의 Group CV 결과를 행 단위로 검증했고, T6R의 R12/R1/R0과 같은 행끼리 비교했다. T6R 실행 결과와 보고서는 변경하지 않았다.

## 2. 회수 증거 검증

| 항목 | 결과 |
|---|---|
| 회수 ZIP `earthquake_geo_group_cv_private_20261014.zip` | SHA `346948cc…2a57e4`, 멤버 77개가 내부 SHA256SUMS와 모두 일치 |
| C geo2/geo1 OOF | `9e58f72b…` / `45360835…`. T6 `config.json`과 T5 공개 요약에 고정된 SHA와 일치 |
| T5 manifest | geo2 `36a40ddf…`, geo1 `1b4616ff…`. StratifiedGroupKFold(seed 42) 재구성과 바이트 단위로 일치 |
| C 모델 10개 | Fold 기록의 SHA 확인 후 재로딩. 학습 Fold 전처리 통계, 검증 행 전체 예측(확률 atol 1e-12)이 OOF와 일치 |
| C Group CV 공개 집계 | 행 단위 OOF로 재계산한 평균·SD가 T6R 보고서에 쓴 공개 집계와 일치(atol 1e-12) |
| T6R 실행 결과 | design seal, 원장 45/45, 실행본·백업 SHA 재확인. 감사 전후 바이트 불변 |
| 데이터 | development.csv / split_manifest.csv SHA가 data contract와 일치하고 불변 |

검증에는 T6 감사의 `reference()`와 `turnover()`, T5의 독립 `arithmetic()`을 **변경 없이** 재사용했다. 감사 코드는 `experiments/geo_ablation_c_group_supplement/audit.py`다.

보조 worker가 공개 기록을 별도로 대조했다. 공개 SHA가 있는 ZIP 파일 약 30개는 모두 일치했다. 다만 다음 두 가지는 공개 기록에 SHA 앵커가 없다.

- 모델 joblib 20개(이번 감사에서 쓴 C 10개 포함)
- ZIP 자체

C 모델은 공개 앵커가 있는 OOF를 1e-12 이내로 재현한다. 따라서 기능적으로는 검증된 것으로 본다.

## 3. Group CV 같은 행 비교 (C 대비, 208,480행)

Fold 우위는 5개 Fold 각각에서 C보다 Macro F1이 높은 Fold의 수다. Δ는 변형 − C다.

| 검증 | 모델 | Fold 우위 | Fold ΔF1 범위 | 정답 전환(C 오류→정답) | 새 오류(C 정답→오류) | 예측 일치 | Δ실제3 Recall | Δ3→1 | Δ3→2 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 2차 지역 | R12 | 0/5 | −0.016~−0.002 | 3,356 | 3,151 | 96.8% | −0.87%p | −22 | +629 |
| 2차 지역 | **R1** | **5/5** | +0.012~+0.027 | 6,699 | 5,533 | 94.0% | −0.23%p | +59 | +102 |
| 2차 지역 | R0 | 0/5 | −0.135~−0.095 | 22,873 | 38,851 | 70.0% | −34.99%p | +174 | +24,241 |
| 1차 지역 | R12 | 0/5 | −0.009~−0.004 | 845 | 824 | 99.2% | −0.41%p | −10 | +296 |
| 1차 지역 | R1 | 4/5 | −0.004~+0.010 | 1,283 | 1,252 | 98.7% | −0.57%p | +89 | +312 |
| 1차 지역 | **R0** | **5/5** | +0.042~+0.079 | 11,753 | 13,830 | 87.4% | +13.20%p | +221 | −9,430 |

등급별 Recall(pooled)은 다음과 같다.

| 검증 | 모델 | Recall 1 | Recall 2 | Recall 3 | Accuracy |
|---|---|---:|---:|---:|---:|
| 2차 지역 | C | 29.2% | 80.6% | 52.1% | 66.13% |
| 2차 지역 | R1 | 35.2% | 80.7% | 51.9% | 66.68% |
| 1차 지역 | C | 25.0% | 96.8% | 0.6% | 57.70% |
| 1차 지역 | R0 | 26.4% | 87.1% | 13.8% | 56.70% |

## 4. 해석 범위

- **미관측 2차 지역의 R1 우위(+0.018)는 대부분 1등급 Recall 상승(29.2%→35.2%)에서 나온다.**
  - 실제 3등급 Recall은 C와 거의 같다(−0.23%p).
  - 3→2 오류는 2,138건이 없어지고 2,240건이 새로 생겼다. 결과적으로 +102건이다.
  - 즉 geo2/geo3을 빼면 중증 과소예측은 줄지 않는다. 대신 경미 피해 구분이 나아진다.
- **미관측 1차 지역의 R0 우위(+0.059 pooled)는 3등급 과소예측 감소에서 나온다.**
  - 3→2가 9,430건 줄었다(제거 9,804, 신규 374).
  - 대가로 C가 맞힌 13,830행이 새로 틀렸다. 정답 전환은 11,753행에 그쳐 Accuracy는 −1.0%p다.
  - 2등급 Recall은 96.8%에서 87.1%로 떨어진다.
  - 3→1은 221건 늘었다.
- 지정 지역 26/20/13과 age995 집단의 같은 행 결과는 [집계 JSON](geo_ablation_c_group_supplement_20261014.json)의 `paired`에 있다. Fold별 ΔF1은 `fold_wins`에 있다.
- T6R §6의 결론은 바뀌지 않는다. 랜덤 CV 성능과 미관측 지역 일반화 사이의 교환관계는 기술적 CV 결과일 뿐이다. 인과·누수 증명, 통계적 유의성, 외부 지진·홀드아웃 성능은 주장하지 않는다. C를 기준선으로 유지한다.

## 5. 재현과 비공개 산출물

```
python -m experiments.geo_ablation_c_group_supplement.audit --data <data> --package <T5 패키지 해제 경로> --zip <회수 ZIP> --t6r <T6R run> --destination <새 경로> --public <새 JSON>
```

- 환경: Python 3.13.5 / scikit-learn 1.8.0. T6R과 같은 동결 venv를 사용했다.
- 비공개 산출물은 로컬 `local_reproduction/T6S_20261014_c_group/`에 있다. 구성은 `t5_private/`(ZIP 해제본), `audit/`, `audit.log`, `unittest.log`다. 원시 데이터·모델·OOF·manifest는 Git에 넣지 않았다.
- 저장소 테스트: 331개 실행, 실패·오류 0, 건너뜀 29.
