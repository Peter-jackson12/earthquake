"""네팔 지진 피해등급: 파생변수·샘플링 실험 v2 (팀 고정 5-fold).

원본: C:\\t_project_1_1\\nepal_interaction_sampling_experiments.py (test_3.ipynb와 같은 코드)

원본과 달라진 점
- 원본 CSV를 seed 42로 직접 분할(building_id 정렬 없음)하던 것을
  preprocess.load_development()와 저장된 5-fold로 교체. 팀 final_holdout은 읽지 않음.
- validation 1회 + 3-fold 두 단계를 같은 5-fold 한 번으로 통합하고,
  기준(원본38 / none) 대비 fold별 짝지은 차이를 함께 저장.
- 학습 fold에 없던 범주는 NaN으로 바꾼 뒤 범주형으로 변환 (Pandas4Warning 제거).
- Macro F1을 labels=[1,2,3]으로 명시.

실행 (저장소 루트에서):
    python tests/t_project_v2/interaction_sampling_v2.py

산출: outputs/t_project_v2/interaction_sampling/ 아래 CSV (gitignore 대상)
"""
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)

REPO = next(p for p in Path(__file__).resolve().parents if (p / "preprocess.py").exists())
sys.path.insert(0, str(REPO))

import preprocess as pp  # noqa: E402

SEED = 42
LABELS = [1, 2, 3]
OUT = REPO / "outputs" / "t_project_v2" / "interaction_sampling"
OUT.mkdir(parents=True, exist_ok=True)

X_base, y, ids, cv = pp.load_development(REPO / "data")
y = y.astype(int)
folds = list(cv.split())
assert X_base.shape == (208480, 38) and len(folds) == 5

# 원본 스크립트의 결합변수 5개 (입력 변수끼리의 Cramer's V로 사전 선정, 정답은 쓰지 않음)
pairs = {
    "geo1_x_mudstone": ("geo_level_1_id", "has_superstructure_mud_mortar_stone"),
    "foundation_x_mudstone": ("foundation_type", "has_superstructure_mud_mortar_stone"),
    "roof_x_rceng": ("roof_type", "has_superstructure_rc_engineered"),
    "groundfloor_x_cementbrick": ("ground_floor_type", "has_superstructure_cement_mortar_brick"),
    "otherfloor_x_mudstone": ("other_floor_type", "has_superstructure_mud_mortar_stone"),
}
X_joint = X_base.copy()
for name, (col1, col2) in pairs.items():
    X_joint[name] = X_joint[col1].astype(str) + "|" + X_joint[col2].astype(str)

# 대·중·소 통합지역. geo_level_3가 상위 지역을 하나씩만 가지므로 geo_level_3와 정보가 같음
X_geo = X_joint.copy()
X_geo["region_combined"] = (
    X_geo["geo_level_1_id"].astype(str) + "_"
    + X_geo["geo_level_2_id"].astype(str) + "_"
    + X_geo["geo_level_3_id"].astype(str)
)
feature_sets = {
    "원본38": X_base,
    "원본+결합5": X_joint,
    "원본+통합지역": X_base.assign(region_combined=X_geo["region_combined"]),
    "원본+결합5+통합지역": X_geo,
}

geo_cols = set(pp.GEO)


def encode_per_train(data, train_ids, valid_ids):
    """학습 fold의 범주 목록만으로 검증 fold를 인코딩. 처음 보는 값은 NaN."""
    train = data.iloc[train_ids].copy()
    valid = data.iloc[valid_ids].copy()
    categorical = [
        c for c in data.columns
        if c in geo_cols or not pd.api.types.is_numeric_dtype(data[c])
    ]
    for c in categorical:
        train[c] = train[c].astype("category")
        known = valid[c].where(valid[c].isin(train[c].cat.categories))
        valid[c] = pd.Categorical(known, categories=train[c].cat.categories)
    return train, valid, categorical


def resample_train(X, y_values, method):
    """학습 fold만 리샘플링. 검증 fold는 원래 분포 유지."""
    if method in {"none", "weight_balanced"}:
        return X, y_values
    if method == "under_full":
        sampler = RandomUnderSampler(random_state=SEED)
    elif method in {"under_partial", "under_partial+weight_balanced"}:
        # 다수인 2등급을 3등급 수까지만 축소, 1등급과 3등급은 유지
        cap = int(y_values.value_counts().loc[3])
        sampler = RandomUnderSampler(sampling_strategy={2: cap}, random_state=SEED)
    elif method == "over_full":
        sampler = RandomOverSampler(random_state=SEED)
    else:
        raise ValueError(f"알 수 없는 샘플링 방식: {method}")
    return sampler.fit_resample(X, y_values)


def evaluate(X, train_ids, valid_ids, feature_name, method, fold):
    start = perf_counter()
    tr, va, cat = encode_per_train(X, train_ids, valid_ids)
    yt, yv = y.iloc[train_ids].copy(), y.iloc[valid_ids].to_numpy()
    tr, yt = resample_train(tr, yt, method)
    model = LGBMClassifier(
        random_state=SEED,
        n_estimators=100,
        n_jobs=4,
        verbosity=-1,
        class_weight="balanced" if "weight_balanced" in method else None,
    )
    model.fit(tr, yt, categorical_feature=cat)
    pred = model.predict(va)
    cm = confusion_matrix(yv, pred, labels=LABELS)
    n3 = int(cm[2].sum())
    return {
        "features": feature_name,
        "sampling": method,
        "fold": fold,
        "train_samples": len(yt),
        "macro_f1": f1_score(yv, pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(yv, pred),
        "recall1": recall_score(yv, pred, labels=[1], average=None, zero_division=0)[0],
        "recall2": recall_score(yv, pred, labels=[2], average=None, zero_division=0)[0],
        "recall3": cm[2, 2] / n3,
        "precision3": precision_score(yv, pred, labels=[3], average=None, zero_division=0)[0],
        "err3to1": cm[2, 0] / n3,
        "err3to2": cm[2, 1] / n3,
        "err3to1_count": int(cm[2, 0]),
        "elapsed_seconds": round(perf_counter() - start, 1),
    }


def run(rows, X, feature_name, method, path):
    for fold, (tr, va) in enumerate(folds):
        rows.append(evaluate(X, tr, va, feature_name, method, fold))
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    scores = [r["macro_f1"] for r in rows if r["features"] == feature_name and r["sampling"] == method]
    print(f"{feature_name:<28} {method:<30} Macro F1 {np.mean(scores):.5f} ± {np.std(scores, ddof=1):.5f}", flush=True)


def summarize(rows, baseline=("원본38", "none")):
    """설정별 평균·표준편차와 기준 대비 fold별 짝지은 차이."""
    table = pd.DataFrame(rows)
    table["config"] = table["features"] + " / " + table["sampling"]
    wide = table.pivot(index="fold", columns="config", values="macro_f1")
    diff = wide.sub(wide[" / ".join(baseline)], axis=0)
    summary = (
        table.groupby(["config", "features", "sampling"])
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            recall1=("recall1", "mean"),
            recall2=("recall2", "mean"),
            recall3=("recall3", "mean"),
            err3to2=("err3to2", "mean"),
            train_samples=("train_samples", "mean"),
        )
        .reset_index()
    )
    summary["diff_vs_base"] = summary["config"].map(diff.mean())
    summary["diff_std"] = summary["config"].map(diff.std())
    summary["wins_vs_base"] = summary["config"].map((diff > 0).sum())
    return summary.sort_values("macro_f1", ascending=False)


if __name__ == "__main__":
    sampling_methods = [
        "none", "weight_balanced", "under_partial", "under_full",
        "over_full", "under_partial+weight_balanced",
    ]

    # 1) 원본38 / 원본+결합5 × 6가지 샘플링
    # 2) 통합지역 추가 여부 (두 주요 방법)
    rows = []
    path = OUT / "cv_feature_sampling_folds.csv"
    for feat_name in ["원본38", "원본+결합5"]:
        for method in sampling_methods:
            run(rows, feature_sets[feat_name], feat_name, method, path)
    for feat_name in ["원본+통합지역", "원본+결합5+통합지역"]:
        for method in ["none", "under_partial"]:
            run(rows, feature_sets[feat_name], feat_name, method, path)

    summary = summarize(rows)
    summary.to_csv(OUT / "cv_feature_sampling_summary.csv", index=False, encoding="utf-8-sig")

    # 3) 결합변수를 1개씩 추가 (기준은 같은 fold의 원본38)
    ablation = [r for r in rows if r["features"] == "원본38" and r["sampling"] in {"none", "under_partial"}]
    n_base = len(ablation)
    path = OUT / "cv_single_feature_ablation_folds.csv"
    for name in pairs:
        one = X_base.assign(**{name: X_joint[name]})
        for method in ["none", "under_partial"]:
            run(ablation, one, "원본+" + name, method, path)

    ablation_summary = summarize(ablation)
    ablation_summary.to_csv(OUT / "cv_single_feature_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # 무결성: 모든 설정이 5개 fold로 평가됐는지
    for table in [pd.DataFrame(rows), pd.DataFrame(ablation[n_base:])]:
        assert table.groupby(["features", "sampling"])["fold"].nunique().eq(5).all()

    pd.set_option("display.width", 200)
    print("\n[파생변수 × 샘플링, 5-fold]")
    print(summary.drop(columns=["config"]).round(5).to_string(index=False))
    print("\n[결합변수 1개씩 추가, 5-fold]")
    print(ablation_summary.drop(columns=["config"]).round(5).to_string(index=False))
    print("\n완료. 팀 final_holdout은 사용하지 않았습니다. 결과:", OUT)
