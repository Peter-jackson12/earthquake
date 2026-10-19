"""
실험 ③ 파생변수 (변수 93개)

- 아래 함수들은 노트북(exp3_softvoting_1.ipynb)과 똑같은 코드입니다. 모델 학습(train_model.py)에 씁니다.
- FeatureBuilder는 학습 데이터로 만든 지역 통계·피해 비율 표를 저장해 두었다가,
  앱에서 새 건물 한 채가 들어와도 바로 같은 변수 93개를 만들어 줍니다.
  (노트북의 make_features에서 검증/테스트 데이터를 처리하는 방식과 같은 결과)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RATE_KEYS = geo_cols + ["material_group", "structure_type", "g1_material", "g2_material", "g3_material"]


# ------------------------------------------------------------------ 노트북과 같은 함수
def add_9_features(data):
    """한 건물(한 행) 안의 정보만 쓰는 변수 -> 누수 걱정 없음"""
    data = data.copy()
    data["is_995"] = (data["age"] == 995).astype(int)
    data["height_area_ratio"] = data["height_percentage"] / data["area_percentage"].clip(lower=1)
    material_cols = [c for c in data.columns if c.startswith("has_superstructure_")]
    data["material_count"] = data[material_cols].sum(axis=1)
    use_cols = [c for c in data.columns if c.startswith("has_secondary_use_")]
    data["secondary_use_count"] = data[use_cols].sum(axis=1)
    data["height_per_floor"] = data["height_percentage"] / data["count_floors_pre_eq"].clip(lower=1)
    data["rc_count"] = (data["has_superstructure_rc_non_engineered"]
                        + data["has_superstructure_rc_engineered"])
    data["mud_count"] = (data["has_superstructure_adobe_mud"]
                         + data["has_superstructure_mud_mortar_stone"]
                         + data["has_superstructure_mud_mortar_brick"])
    data["foundation_roof"] = data["foundation_type"] + "_" + data["roof_type"]
    data["ground_other_floor"] = data["ground_floor_type"] + "_" + data["other_floor_type"]
    data["foundation_ground"] = data["foundation_type"] + "_" + data["ground_floor_type"]
    data["surface_condition"] = data["land_surface_condition"] + "_" + data["ground_floor_type"]
    data["has_mud"] = (data["mud_count"] > 0).astype(int)
    data["has_rc"] = (data["rc_count"] > 0).astype(int)
    return data


def add_region_age(train, other):
    """지역별 평균 나이 - 학습 데이터에서만 계산"""
    train, other = train.copy(), other.copy()
    normal_age = train[train["age"] != 995]
    overall_mean = normal_age["age"].mean()
    for geo in geo_cols:
        age_mean = normal_age.groupby(geo)["age"].mean()
        train[geo + "_age_mean"] = train[geo].map(age_mean).fillna(overall_mean)
        other[geo + "_age_mean"] = other[geo].map(age_mean).fillna(overall_mean)
    return train, other


def add_more_features(data):
    data = data.copy()
    data["cement_mortar_count"] = (data["has_superstructure_cement_mortar_stone"]
                                   + data["has_superstructure_cement_mortar_brick"])
    data["wood_bamboo_count"] = (data["has_superstructure_timber"]
                                 + data["has_superstructure_bamboo"])
    data["n_weak_material"] = (data["has_superstructure_adobe_mud"]
                               + data["has_superstructure_mud_mortar_stone"]
                               + data["has_superstructure_stone_flag"]
                               + data["has_superstructure_mud_mortar_brick"])
    data["material_group"] = "other"
    data.loc[data["wood_bamboo_count"] > 0, "material_group"] = "timber_bamboo"
    data.loc[data["n_weak_material"] > 0, "material_group"] = "mud_stone"
    data.loc[data["cement_mortar_count"] > 0, "material_group"] = "cement"
    data.loc[data["rc_count"] > 0, "material_group"] = "RC"
    data["structure_type"] = (data["foundation_type"] + "_" + data["roof_type"]
                              + "_" + data["ground_floor_type"])
    data["floor_band"] = "3F+"
    data.loc[data["count_floors_pre_eq"] == 1, "floor_band"] = "1F"
    data.loc[data["count_floors_pre_eq"] == 2, "floor_band"] = "2F"
    is_weak = (data["material_group"] == "mud_stone") & (data["age"] != 995)
    data["weak_age"] = np.where(is_weak, data["age"], 0)
    return data


def add_region_profile(train, other):
    """지역별 건물 특징 - 정답은 쓰지 않고 학습 데이터로만 계산"""
    train, other = train.copy(), other.copy()
    train["is_weak"] = (train["n_weak_material"] > 0).astype(int)
    for i, geo in enumerate(geo_cols, start=1):
        summary = train.groupby(geo).agg(
            building_count=("age", "size"),
            weak_ratio=("is_weak", "mean"),
            rc_ratio=("has_rc", "mean"),
        )
        for data in [train, other]:
            data[f"g{i}_building_count"] = data[geo].map(summary["building_count"]).fillna(0)
            data[f"g{i}_weak_ratio"] = data[geo].map(summary["weak_ratio"]).fillna(train["is_weak"].mean())
            data[f"g{i}_rc_ratio"] = data[geo].map(summary["rc_ratio"]).fillna(train["has_rc"].mean())
            data[f"g{i}_age_gap"] = np.where(data["age"] == 995, 0,
                                             data["age"] - data[f"{geo}_age_mean"])
    train = train.drop(columns=["is_weak"])
    return train, other


def make_rate_table(keys, y, smooth=20):
    """그룹별 1등급 비율(low_rate), 3등급 비율(high_rate), smoothing 적용"""
    temp = pd.DataFrame({
        "key": keys.to_numpy(),
        "low": (y.to_numpy() == 0).astype(int),
        "high": (y.to_numpy() == 2).astype(int),
    })
    table = temp.groupby("key").agg(count=("low", "size"),
                                    low_sum=("low", "sum"),
                                    high_sum=("high", "sum"))
    table["low_rate"] = (table["low_sum"] + smooth * temp["low"].mean()) / (table["count"] + smooth)
    table["high_rate"] = (table["high_sum"] + smooth * temp["high"].mean()) / (table["count"] + smooth)
    return table


def add_damage_rates(train, other, y_train):
    """그룹별 피해 등급 비율 (5-fold로 누수 방지)"""
    train, other = train.copy(), other.copy()
    y_train = y_train.reset_index(drop=True)
    for data in [train, other]:
        for i, geo in enumerate(geo_cols, start=1):
            data[f"g{i}_material"] = data[geo].astype(str) + "_" + data["material_group"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for key in RATE_KEYS:
        low_col, high_col = key + "_low_rate", key + "_high_rate"
        train[low_col] = np.nan
        train[high_col] = np.nan
        for fit_idx, oof_idx in folds.split(train, y_train):
            fit_y = y_train.iloc[fit_idx]
            table = make_rate_table(train[key].iloc[fit_idx], fit_y)
            part = train[key].iloc[oof_idx]
            train.loc[oof_idx, low_col] = part.map(table["low_rate"]).fillna((fit_y == 0).mean()).to_numpy()
            train.loc[oof_idx, high_col] = part.map(table["high_rate"]).fillna((fit_y == 2).mean()).to_numpy()
        table = make_rate_table(train[key], y_train)
        other[low_col] = other[key].map(table["low_rate"]).fillna((y_train == 0).mean())
        other[high_col] = other[key].map(table["high_rate"]).fillna((y_train == 2).mean())
    for data in [train, other]:
        for i in [2, 3]:
            data[f"g{i}_material_high_gap"] = data[f"g{i}_material_high_rate"] - data["material_group_high_rate"]
            data[f"g{i}_material_low_gap"] = data[f"g{i}_material_low_rate"] - data["material_group_low_rate"]
    drop_cols = ["g1_material", "g2_material", "g3_material"]
    return train.drop(columns=drop_cols), other.drop(columns=drop_cols)


def to_category(train, other):
    """글자 변수와 지역 ID를 범주형으로 (학습에 없던 값은 결측)"""
    train, other = train.copy(), other.copy()
    cat_cols = train.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = cat_cols + [c for c in geo_cols if c not in cat_cols]
    for col in cat_cols:
        train[col] = train[col].astype("category")
        known = train[col].cat.categories
        other[col] = pd.Categorical(other[col].where(other[col].isin(known)), categories=known)
    assert list(train.columns) == list(other.columns), "학습/검증 변수 구성이 다릅니다."
    return train, other


def make_features(train_raw, other_raw, y_train):
    train = add_9_features(train_raw)
    other = add_9_features(other_raw)
    train, other = add_region_age(train, other)
    train = add_more_features(train)
    other = add_more_features(other)
    train, other = add_region_profile(train, other)
    train, other = add_damage_rates(train, other, y_train)
    return to_category(train, other)


# ------------------------------------------------------------------ 모델 입력 변환
def to_codes(data, missing_as_nan=False):
    """범주형 -> 숫자 코드 (처음 보는 값은 -1 또는 NaN)"""
    data = data.copy()
    for col in data.select_dtypes(include="category").columns:
        codes = data[col].cat.codes.astype(float)
        if missing_as_nan:
            codes[codes < 0] = np.nan
        data[col] = codes
    return data


def hgb_categorical_mask(data):
    """HGB용: 범주 255개 이하인 범주형 변수만 True"""
    return [(str(data[c].dtype) == "category") and (data[c].cat.categories.size <= 255)
            for c in data.columns]


# ------------------------------------------------------------------ 새 건물용 변환기
class FeatureBuilder:
    """학습 데이터로 표를 만들어 두고(fit), 새 건물에 같은 변수를 붙임(transform).
    결과는 make_features(학습, 새 데이터)의 '새 데이터' 쪽과 같습니다."""

    def fit(self, X_raw, y, reference_columns):
        y = y.reset_index(drop=True)
        t = add_9_features(X_raw.reset_index(drop=True))

        # 지역별 평균 나이
        normal_age = t[t["age"] != 995]
        self.overall_age = normal_age["age"].mean()
        self.age_mean = {geo: normal_age.groupby(geo)["age"].mean() for geo in geo_cols}
        for geo in geo_cols:
            t[geo + "_age_mean"] = t[geo].map(self.age_mean[geo]).fillna(self.overall_age)

        t = add_more_features(t)

        # 지역별 건물 특징
        is_weak = (t["n_weak_material"] > 0).astype(int)
        self.weak_mean, self.rc_mean = is_weak.mean(), t["has_rc"].mean()
        self.profile = {}
        for geo in geo_cols:
            self.profile[geo] = pd.DataFrame({"age": t["age"], "is_weak": is_weak,
                                              "has_rc": t["has_rc"], geo: t[geo]}).groupby(geo).agg(
                building_count=("age", "size"), weak_ratio=("is_weak", "mean"), rc_ratio=("has_rc", "mean"))

        # 피해 등급 비율 표 (학습 데이터 전체)
        for i, geo in enumerate(geo_cols, start=1):
            t[f"g{i}_material"] = t[geo].astype(str) + "_" + t["material_group"]
        self.rate_tables = {key: make_rate_table(t[key], y) for key in RATE_KEYS}
        self.low_default, self.high_default = (y == 0).mean(), (y == 2).mean()

        # 범주형 목록과 변수 순서 (학습에 쓴 데이터 기준)
        self.columns = list(reference_columns.columns)
        self.categories = {c: reference_columns[c].cat.categories
                           for c in reference_columns.columns if str(reference_columns[c].dtype) == "category"}
        return self

    def transform(self, X_raw):
        d = add_9_features(X_raw.reset_index(drop=True))
        for geo in geo_cols:
            d[geo + "_age_mean"] = d[geo].map(self.age_mean[geo]).fillna(self.overall_age)
        d = add_more_features(d)
        for i, geo in enumerate(geo_cols, start=1):
            p = self.profile[geo]
            d[f"g{i}_building_count"] = d[geo].map(p["building_count"]).fillna(0)
            d[f"g{i}_weak_ratio"] = d[geo].map(p["weak_ratio"]).fillna(self.weak_mean)
            d[f"g{i}_rc_ratio"] = d[geo].map(p["rc_ratio"]).fillna(self.rc_mean)
            d[f"g{i}_age_gap"] = np.where(d["age"] == 995, 0, d["age"] - d[f"{geo}_age_mean"])
        for i, geo in enumerate(geo_cols, start=1):
            d[f"g{i}_material"] = d[geo].astype(str) + "_" + d["material_group"]
        for key in RATE_KEYS:
            table = self.rate_tables[key]
            d[key + "_low_rate"] = d[key].map(table["low_rate"]).fillna(self.low_default)
            d[key + "_high_rate"] = d[key].map(table["high_rate"]).fillna(self.high_default)
        for i in [2, 3]:
            d[f"g{i}_material_high_gap"] = d[f"g{i}_material_high_rate"] - d["material_group_high_rate"]
            d[f"g{i}_material_low_gap"] = d[f"g{i}_material_low_rate"] - d["material_group_low_rate"]
        d = d.drop(columns=["g1_material", "g2_material", "g3_material"])
        for col, cats in self.categories.items():
            d[col] = pd.Categorical(d[col].where(d[col].isin(cats)), categories=cats)
        return d[self.columns]
