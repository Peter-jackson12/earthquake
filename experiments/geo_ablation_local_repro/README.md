# T6R — T6 지역 변수 Ablation 로컬 독립 재현

원래 T6 결과와 **별도의 재현 실험**이다(실험 ID `T6R_geo_ablation_local_repro_v1`). 결과·감사는 [보고서](../../reports/GEO_ABLATION_LOCAL_REPRO_20261014.md)와 [집계 JSON](../../reports/geo_ablation_local_repro_20261014.json)을 본다. 모델 정의는 `experiments/geo_ablation/run.py`의 `make_pipeline`을 변경 없이 사용한다.

```bash
# 1) T5 manifest 재생성(새 비공개 경로) 후 SHA가 T6 config와 일치해야 한다
python -m experiments.geo_group_cv.run prepare --data <DATA> --output <PRIV>/t5_manifest_regen
# 2) 학습 전 봉인
python -m experiments.geo_ablation_local_repro.run prepare --data <DATA> --output <PRIV>/run --manifests <PRIV>/t5_manifest_regen --c-stratified-oof <C_STRATIFIED_OOF> --code-commit <FULL_SHA>
# 3) Fold별 체크포인트 학습. 중단 후 같은 명령으로 재개하며, 완료 Fold는 해시만 확인하고 재학습하지 않는다
python -m experiments.geo_ablation_local_repro.run fit-all --data <DATA> --output <PRIV>/run
python -m experiments.geo_ablation_local_repro.run status --output <PRIV>/run
# 4) 독립 감사(분류기 학습 없음, 새 경로에만 기록)
python -m experiments.geo_ablation_local_repro.audit --data <DATA> --output <PRIV>/run --c-stratified-oof <C_STRATIFIED_OOF> --destination <PRIV>/audit --public <PRIV>/summary.json
```

`<PRIV>`는 Git 밖의 비공개 경로다. 모델·OOF·manifest·로그를 공개 Git에 올리지 않는다. 홀드아웃은 읽지 않는다.
