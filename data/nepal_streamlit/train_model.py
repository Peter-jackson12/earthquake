"""
최종 모델 학습 스크립트 (앱의 '피해 등급 예측' 페이지에서 사용)

노트북 exp3_softvoting_1.ipynb의 10번 '최종 테스트'와 같은 과정입니다.
- 실험 ③ 변수 93개
- HGB + LightGBM soft voting, class_weight 1.9 / 1.0 / 1.3
- 학습+검증(85%)으로 학습, 트리 수는 노트북에서 정한 값(LightGBM 368, HGB 250)으로 고정
- 테스트(15%)로 한 번 평가

실행:  python train_model.py      (2~5분 정도 걸립니다)
결과:  models/final_model.joblib, models/test_metrics.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from features import (SEED, FeatureBuilder, hgb_categorical_mask, make_features,
                      to_codes)

DATA_PATH = Path(__file__).parent / "nepal_building_damage.csv"
MODEL_DIR = Path(__file__).parent / "models"

CLASS_WEIGHT = {0: 1.9, 1: 1.0, 2: 1.3}
LGBM_ITER, HGB_ITER = 368, 250
LGBM_PARAMS = {
    "objective": "multiclass", "learning_rate": 0.03, "num_leaves": 127,
    "min_child_samples": 40, "subsample": 0.85, "subsample_freq": 1,
    "colsample_bytree": 0.7, "reg_lambda": 10, "cat_smooth": 60, "cat_l2": 30,
    "min_data_per_group": 200, "random_state": SEED, "n_jobs": -1, "verbosity": -1,
}
HGB_PARAMS = {
    "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 40,
    "l2_regularization": 5.0, "random_state": SEED,
}


def split_data(df):
    """노트북과 같은 70 / 15 / 15 분할"""
    y = df["damage_grade"] - 1
    X = df.drop(columns=["building_id", "damage_grade"])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=SEED, stratify=y_temp)
    X_dev = pd.concat([X_train, X_val], ignore_index=True)
    y_dev = pd.concat([y_train, y_val], ignore_index=True)
    return X_dev, y_dev, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def main(log=print):
    start = time.time()
    df = pd.read_csv(DATA_PATH)
    X_dev, y_dev, X_test, y_test = split_data(df)
    log(f"데이터: 학습+검증 {len(X_dev):,}건 / 테스트 {len(X_test):,}건")

    final_train, final_test = make_features(X_dev, X_test, y_dev)
    log(f"변수 만들기 완료: {final_train.shape[1]}개")

    builder = FeatureBuilder().fit(X_dev, y_dev, final_train)
    check = builder.transform(X_test)
    num = check.select_dtypes("number").columns
    assert np.allclose(check[num].to_numpy(float), final_test[num].to_numpy(float), equal_nan=True), \
        "FeatureBuilder 결과가 노트북 방식과 다릅니다."
    log("새 건물용 변환기 확인 완료 (노트북 방식과 동일)")

    w_dev = y_dev.map(CLASS_WEIGHT).to_numpy()

    lgbm = LGBMClassifier(**LGBM_PARAMS, n_estimators=LGBM_ITER)
    lgbm.fit(final_train, y_dev, sample_weight=w_dev)
    log("LightGBM 학습 완료")

    hgb_mask = hgb_categorical_mask(final_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS, max_iter=HGB_ITER,
                                         categorical_features=hgb_mask, early_stopping=False)
    hgb.fit(to_codes(final_train, missing_as_nan=True), y_dev, sample_weight=w_dev)
    log("HGB 학습 완료")

    proba = (lgbm.predict_proba(final_test) + hgb.predict_proba(to_codes(final_test, missing_as_nan=True))) / 2
    pred = proba.argmax(axis=1)
    metrics = {
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="macro"),
        "recall": recall_score(y_test, pred, average="macro"),
        "recall_by_grade": recall_score(y_test, pred, average=None).tolist(),
        "precision_by_grade": precision_score(y_test, pred, average=None).tolist(),
        "f1_by_grade": f1_score(y_test, pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "train_seconds": round(time.time() - start),
    }
    log(f"테스트 Macro F1 {metrics['macro_f1']:.4f} / Accuracy {metrics['accuracy']:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"lgbm": lgbm, "hgb": hgb, "builder": builder, "class_weight": CLASS_WEIGHT},
                MODEL_DIR / "final_model.joblib", compress=3)
    (MODEL_DIR / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    log(f"저장 완료: {MODEL_DIR} ({time.time() - start:.0f}초)")
    return metrics


if __name__ == "__main__":
    main()
