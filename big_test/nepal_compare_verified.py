"""네팔 피해 등급 예측: 팀원 파생변수 기반 누수 방지 모델 비교.

실행: uv run python nepal_compare_safe.py
원본 CSV: 실행 인수로 파일을 지정하거나 같은 폴더에 nepal_building_damage(2).csv 또는 nepal_building_damage.csv 배치
주의: 검증 점수로 모델 선택 후, 테스트는 선택된 한 모델에 한 번만 사용.
"""
from pathlib import Path
import hashlib
import gc

# %% 0. 설정과 라이브러리
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

SEED = 42
GRADES = [1, 2, 3]
GEO = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
CATEGORICAL = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
]
CONCRETE = [
    "has_superstructure_cement_mortar_stone", "has_superstructure_cement_mortar_brick",
    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
]
MUD = [
    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
    "has_superstructure_mud_mortar_brick",
]
CROSSES = ["foundation_roof", "ground_other_floor", "foundation_ground", "surface_condition"]


# %% 1. 팀원 파생변수
def make_team_features(train, other):
    """팀원 파생변수 14개: 정답(y) 없이 생성. 지역 연식 평균은 train 기준."""
    a, b = train.copy(), other.copy()
    all_material = [c for c in train if c.startswith("has_superstructure_")]
    # 전체 2차용도 여부(has_secondary_use)는 제외하고 세부용도만 센다.
    uses = [c for c in train if c.startswith("has_secondary_use_")]
    if len(all_material) != 11 or len(uses) != 10:
        raise ValueError("상부 구조 11개/세부 부가용도 10개 열을 확인하세요.")
    for x in (a, b):
        x["is_995"] = (x["age"] == 995).astype(int)
        x["height_area_ratio"] = x["height_percentage"] / x["area_percentage"].clip(lower=1)
        x["material_count"] = x[all_material].sum(axis=1)
        x["secondary_use_count"] = x[uses].sum(axis=1)
        x["height_per_floor"] = x["height_percentage"] / x["count_floors_pre_eq"].clip(lower=1)
        x["concrete_count"] = x[CONCRETE].sum(axis=1)
        x["mud_count"] = x[MUD].sum(axis=1)
        x["foundation_roof"] = x["foundation_type"].astype(str) + "_" + x["roof_type"].astype(str)
        x["ground_other_floor"] = x["ground_floor_type"].astype(str) + "_" + x["other_floor_type"].astype(str)
        x["foundation_ground"] = x["foundation_type"].astype(str) + "_" + x["ground_floor_type"].astype(str)
        x["surface_condition"] = x["land_surface_condition"].astype(str) + "_" + x["ground_floor_type"].astype(str)

    # 995는 연식 평균 계산에서 제외. 예측할 행의 연식 자체는 그대로 둔다.
    known = train.loc[train["age"] != 995]
    if known.empty:
        raise ValueError("995가 아닌 age 값이 없어 지역 평균을 구할 수 없습니다.")
    fallback = known["age"].mean()
    for geo in GEO:
        avg = known.groupby(geo)["age"].mean()
        for x in (a, b):
            x[f"{geo}_age_mean"] = x[geo].map(avg).fillna(fallback)
    return a, b


# %% 2. 지역별 타깃 인코딩(OOF로 누수 방지)
def te_fit_apply(fit_x, fit_y, apply_x, col, smooth=20):
    """fit 쪽의 피해등급 비율로만 지역별 1/2/3등급 인코딩을 만든다."""
    tmp = pd.DataFrame({"region": fit_x[col].to_numpy(), "grade": fit_y.to_numpy()})
    counts = pd.crosstab(tmp["region"], tmp["grade"]).reindex(columns=GRADES, fill_value=0)
    prior = fit_y.value_counts(normalize=True).reindex(GRADES, fill_value=0).to_numpy()
    means = (counts + smooth * prior) / (counts.sum(axis=1).to_numpy()[:, None] + smooth)
    # 처음 보는 지역은 fit 쪽 전체 등급 비율로 대체한다.
    return pd.DataFrame({
        f"{col}_te_{g}": apply_x[col].map(means[g]).fillna(prior[i]).to_numpy(dtype=np.float32)
        for i, g in enumerate(GRADES)
    })


def make_target_encoding(train, y_train, groups_train, other):
    """Train 인코딩은 그룹 단위 5-fold OOF, Valid/Test는 Train 전체로 변환."""
    # 같은 입력값인 건물들이 OOF의 양편에 나뉘지 않게 그룹별 fold를 만든다.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(train, y_train, groups_train))
    tr_parts, other_parts = [], []
    for col in GEO:
        keys = [f"{col}_te_{g}" for g in GRADES]
        oof = pd.DataFrame(index=np.arange(len(train)), columns=keys, dtype=np.float32)
        for fit_idx, hold_idx in folds:
            converted = te_fit_apply(
                train.iloc[fit_idx], y_train.iloc[fit_idx], train.iloc[hold_idx], col
            )
            oof.iloc[hold_idx] = converted.to_numpy()
        if oof.isna().any().any():
            raise ValueError("OOF 타깃 인코딩에 결측값이 발생했습니다.")
        tr_parts.append(oof)
        other_parts.append(te_fit_apply(train, y_train, other, col))
    return pd.concat(tr_parts, axis=1), pd.concat(other_parts, axis=1)


def make_matrices(train, y_train, groups_train, other):
    """두 실험 모두 같은 지역 인코딩을 사용하도록 원본/팀원 피처를 준비한다."""
    tr_te, ot_te = make_target_encoding(train, y_train, groups_train, other)
    team_tr, team_ot = make_team_features(train, other)
    matrices = {}
    for name, a, b in [("raw", train, other), ("team", team_tr, team_ot)]:
        # 지역 숫자코드를 크기(거리)가 있는 연속형 수치로 취급하지 않는다.
        a = pd.concat([a.drop(columns=GEO).reset_index(drop=True), tr_te], axis=1)
        b = pd.concat([b.drop(columns=GEO).reset_index(drop=True), ot_te], axis=1)
        cats = CATEGORICAL + (CROSSES if name == "team" else [])
        numeric = [col for col in a.columns if col not in cats]
        processor = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cats),
            ("numbers", "passthrough", numeric),
        ])
        Xtr = processor.fit_transform(a).astype(np.float32)
        Xot = processor.transform(b).astype(np.float32)
        if not (np.isfinite(Xtr).all() and np.isfinite(Xot).all()):
            raise ValueError(f"{name}: 인코딩 이후 NaN/무한대가 존재합니다.")
        matrices[name] = (Xtr, Xot)
        print(f"  {name}: 모델 입력 {Xtr.shape[1]}개 피처")
    return matrices


# %% 3. 모델 후보: 같은 검증셋에서 비교
def candidate_models():
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=5, early_stopping=False,
        random_state=SEED,
    )
    rf = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2, max_features=0.8,
        random_state=SEED, n_jobs=-1,
    )
    # 가중치 모델은 강사 보고서의 HGB 후보 가중치를 참고한 사전 고정 실험.
    return {
        "HGB_기본_지역TE": ("raw", hgb, False),
        "HGB_팀원파생_지역TE": ("team", clone(hgb), False),
        "HGB_팀원파생_가중치": ("team", clone(hgb), True),
        "RF_팀원파생_지역TE": ("team", rf, False),
    }


def train_model(model_name, matrices, y_train):
    feature_set, estimator, weighted = candidate_models()[model_name]
    model = clone(estimator)
    Xtr = matrices[feature_set][0]
    if weighted:
        weights = y_train.map({1: 1.7, 2: 1.0, 3: 1.05}).to_numpy()
        model.fit(Xtr, y_train, sample_weight=weights)
    else:
        model.fit(Xtr, y_train)
    return model, feature_set


def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    actual_three = y_true == 3
    n_three = actual_three.sum()
    return {
        "Macro_F1": f1_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall_1": recall_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)[0],
        "Recall_3": recall_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)[2],
        "3_to_1_count": int((actual_three & (y_pred == 1)).sum()),
        "3_to_1_rate": float((actual_three & (y_pred == 1)).sum() / n_three) if n_three else 0.0,
        "3_underprediction_rate": float((actual_three & (y_pred < 3)).sum() / n_three) if n_three else 0.0,
    }


# %% 4. 중복 방지 그룹 분할
def split_without_twins(raw):
    """building_id 이외의 모든 입력이 동일한 건물은 같은 그룹으로 묶는다."""
    X = raw.drop(columns=["building_id", "damage_grade"])
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    development, test = next(outer.split(X, raw["damage_grade"], groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED + 1)
    tr_local, va_local = next(inner.split(X.iloc[development], raw["damage_grade"].iloc[development], groups[development]))
    tr_idx, va_idx = development[tr_local], development[va_local]
    # 검증 1: 동일 피처 그룹이 서로 다른 세트에 섞이지 않았는지 확인
    for a, b in [(tr_idx, va_idx), (tr_idx, test), (va_idx, test)]:
        assert set(groups[a]).isdisjoint(groups[b]), "중복 입력값의 세트 간 겹침 발견"
    # 검증 2: 행 손실/중복/누락 없는지 확인
    assert len(set(tr_idx) | set(va_idx) | set(test)) == len(raw)
    print("전체:", len(raw), "/ Train:", len(tr_idx), "/ Valid:", len(va_idx), "/ Test:", len(test))
    for label, take in [("Train", tr_idx), ("Valid", va_idx), ("Test", test)]:
        rates = raw["damage_grade"].iloc[take].value_counts(normalize=True).reindex(GRADES, fill_value=0)
        print(f"{label} 등급별 비율(%):", (rates * 100).round(2).to_dict())
        if (rates == 0).any():
            raise ValueError(f"{label}에 누락된 damage_grade가 있습니다.")
    return tr_idx, va_idx, test, groups


# %% 5. 전체 실행: Validation 선택 -> Test 최종 확인
def run(csv_path):
    raw = pd.read_csv(csv_path)
    with open(csv_path, "rb") as f:
        file_hash = hashlib.file_digest(f, "sha256").hexdigest()
    print("원본 파일 SHA256:", file_hash)
    # 강사 보고서의 입력 파일 해시 (동일 CSV인지 확인하는 참고값)
    instructor_hash = "87abd2bc2b05d3e687bd176959967fd38334fff607e6afeb2d20d0d899c13e97"
    if file_hash != instructor_hash:
        print("참고: 강사 보고서의 CSV 해시와 다릅니다. 동일 데이터 여부를 확인하세요.")
    required = {"building_id", "damage_grade", "age", "height_percentage", "area_percentage", "count_floors_pre_eq"}
    required.update(GEO + CATEGORICAL + CONCRETE + MUD)
    required.update(["has_secondary_use"])
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"원본 CSV에서 필수 컬럼을 찾지 못했습니다: {missing}")
    if raw["building_id"].duplicated().any():
        raise ValueError("building_id 중복: 동일 건물인지 원본 점검이 필요합니다.")
    if raw.isna().any().any():
        raise ValueError("결측값 존재: 원천 자료를 먼저 점검하세요.")
    if set(raw["damage_grade"].unique()) != set(GRADES):
        raise ValueError("damage_grade는 1, 2, 3이어야 합니다.")
    # '동일 속성 + 다른 damage_grade'도 지우지 않고 동일 그룹에 함께 유지한다.
    fingerprints = pd.util.hash_pandas_object(raw.drop(columns=["building_id", "damage_grade"]), index=False)
    print("원본 행·열:", raw.shape)
    if raw.shape != (260601, 40):
        print("참고: 강사 보고서의 260,601행 × 40열과 차이가 있습니다.")
    print("동일 피처 그룹에 속한 추가 행:", int(len(raw) - fingerprints.nunique()))
    conflicting = raw.groupby(fingerprints)["damage_grade"].nunique().gt(1).sum()
    print("동일 피처인데 타깃이 다른 그룹:", int(conflicting), "(삭제하지 않고 한 분할에 유지)")
    # 이때 동일한 피처만으로 실제 중복 수집 여부를 결정할 수는 없다.
    tr, va, te, groups = split_without_twins(raw)
    features = raw.drop(columns=["building_id", "damage_grade"])
    y = raw["damage_grade"]
    X_train, X_val = features.iloc[tr].reset_index(drop=True), features.iloc[va].reset_index(drop=True)
    y_train, y_val = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True)

    print("\n[1] Train 기준 전처리와 검증 성능 비교")
    mats = make_matrices(X_train, y_train, groups[tr], X_val)
    summary = []
    for name in candidate_models():
        model, kind = train_model(name, mats, y_train)
        pred = model.predict(mats[kind][1])
        summary.append({"Model": name, **score(y_val, pred)})
        print(name, ": Macro F1", round(summary[-1]["Macro_F1"], 5))
        del model
    validation = pd.DataFrame(summary).sort_values("Macro_F1", ascending=False)
    validation.to_csv("nepal_validation_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n검증 순위 (Test를 보지 않고 선택):\n", validation.to_string(index=False))
    selected = validation.iloc[0]["Model"]
    print("\n최종 선택:", selected)
    del mats
    gc.collect()

    print("\n[2] Train+Valid로 새로 학습 -> 선택 모델의 Test 평가 딱 한 번")
    # train/valid에 같은 피처인 그룹이 겹치지 않았으므로 그대로 합쳐 재학습 가능.
    dev = np.concatenate([tr, va])
    X_dev, X_test = features.iloc[dev].reset_index(drop=True), features.iloc[te].reset_index(drop=True)
    y_dev, y_test = y.iloc[dev].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
    mats_final = make_matrices(X_dev, y_dev, groups[dev], X_test)
    final_model, kind = train_model(selected, mats_final, y_dev)
    test_pred = final_model.predict(mats_final[kind][1])
    final = {"Model": selected, **score(y_test, test_pred)}
    pd.DataFrame([final]).to_csv("nepal_final_test.csv", index=False, encoding="utf-8-sig")
    confusion = pd.DataFrame(confusion_matrix(y_test, test_pred, labels=GRADES),
                             index=["actual_1", "actual_2", "actual_3"],
                             columns=["pred_1", "pred_2", "pred_3"])
    confusion.to_csv("nepal_final_confusion.csv", encoding="utf-8-sig")
    print("최종 Test:", final)
    print("\nTest 등급별 평가:\n", classification_report(y_test, test_pred, labels=GRADES, zero_division=0))
    print("\n혼동행렬:\n", confusion)
    print("\n저장 완료: nepal_validation_comparison.csv, nepal_final_test.csv, nepal_final_confusion.csv")
    return validation, final


# %% 6. 실행 시작점
if __name__ == "__main__":
    import sys
    # VS Code 터미널 실행: uv run python nepal_compare_verified.py "nepal_building_damage(2).csv"
    folder = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1]).expanduser()
    else:
        filenames = ["nepal_building_damage(2).csv", "nepal_building_damage.csv"]
        data_path = next((folder / name for name in filenames if (folder / name).exists()), None)
    if data_path is None or not data_path.exists():
        raise FileNotFoundError("원본 CSV를 같은 폴더에 넣거나 실행 인수로 파일 경로를 전달하세요.")
    run(data_path)
