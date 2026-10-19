"""노트북 실행 결과 (앱에서 표·그래프로 보여 주는 값)
출처: 02_baseline, 03-01/03-02/exp3 노트북, 튜닝 노트북, exp3_softvoting_1"""
import pandas as pd

GRADE_LABELS = ["1등급", "2등급", "3등급"]
GRADE_COLORS = {"1등급": "#7E9C8F", "2등급": "#D4A03A", "3등급": "#B4432F"}

# 02_baseline: 원본 38개 변수, 원핫 인코딩, 불균형 처리 방법 비교 (5-Fold CV + 검증)
BASELINE = pd.DataFrame([
    ["CatBoost", "SMOTENC", 0.7166, 0.6514, 0.6651, 0.6671],
    ["CatBoost", "SMOTE", 0.7179, 0.6459, 0.6621, 0.6654],
    ["XGBoost", "SMOTENC", 0.7149, 0.6434, 0.6603, 0.6627],
    ["XGBoost", "SMOTE", 0.7154, 0.6387, 0.6592, 0.6610],
    ["RandomForest", "Class Weight", 0.6930, 0.6786, 0.6555, 0.6609],
    ["CatBoost", "없음", 0.7245, 0.6275, 0.6568, 0.6593],
    ["RandomForest", "SMOTENC", 0.7053, 0.6440, 0.6520, 0.6561],
    ["XGBoost", "없음", 0.7201, 0.6248, 0.6603, 0.6551],
    ["RandomForest", "없음", 0.7099, 0.6260, 0.6484, 0.6503],
    ["LightGBM", "SMOTENC", 0.6985, 0.6361, 0.6482, 0.6493],
    ["RandomForest", "SMOTE", 0.7021, 0.6254, 0.6460, 0.6454],
    ["LightGBM", "SMOTE", 0.6990, 0.6273, 0.6451, 0.6453],
    ["XGBoost", "Class Weight", 0.6642, 0.7192, 0.6401, 0.6424],
    ["LightGBM", "없음", 0.7108, 0.6108, 0.6421, 0.6416],
    ["CatBoost", "Class Weight", 0.6600, 0.7188, 0.6352, 0.6390],
    ["XGBoost", "Under Sampling", 0.6436, 0.7088, 0.6202, 0.6230],
    ["LightGBM", "Class Weight", 0.6397, 0.7000, 0.6170, 0.6181],
    ["CatBoost", "Under Sampling", 0.6377, 0.7042, 0.6143, 0.6172],
    ["LightGBM", "Under Sampling", 0.6282, 0.6939, 0.6083, 0.6076],
    ["RandomForest", "Under Sampling", 0.6205, 0.6849, 0.5955, 0.6015],
], columns=["모델", "불균형 처리", "검증 Accuracy", "검증 Recall", "CV F1 평균", "검증 Macro F1"])

# 실험 ①②③: LightGBM 같은 설정, 등급 배수 조정
EXPERIMENTS = pd.DataFrame([
    ["① 기존 파생변수", 54, 0.7084, 0.7182, "1.7 / 1.0 / 1.5", 799],
    ["② ① + 추가 변수", 61, 0.7085, 0.7178, "1.8 / 1.0 / 1.5", 799],
    ["③ ② + 지역 정보", 93, 0.7067, 0.7184, "1.8 / 1.0 / 1.3", 386],
], columns=["실험", "변수 수", "배수 적용 전 검증 F1", "배수 적용 후 검증 F1", "최적 배수 (1/2/3)", "트리 수"])

EXPERIMENT_FEATURES = {
    "① 기존 파생변수 (+16개)": [
        ("is_995", "건물 나이가 995(이상값)인지 여부"),
        ("height_area_ratio", "높이 / 면적"),
        ("material_count", "사용한 구조 재료 개수"),
        ("secondary_use_count", "사용한 부가용도 개수"),
        ("height_per_floor", "높이 / 층 수"),
        ("rc_count, mud_count", "철근콘크리트·흙 계열 재료 개수"),
        ("has_rc, has_mud", "철근콘크리트·흙 재료 사용 여부"),
        ("foundation_roof 외 3개", "기초·지붕·바닥·지표면 조합"),
        ("geo_level_1/2/3_id_age_mean", "지역별 평균 나이 (학습 데이터 기준)"),
    ],
    "② 추가 파생변수 (+7개)": [
        ("cement_mortar_count", "시멘트 계열 재료 개수"),
        ("wood_bamboo_count", "목재·대나무 개수"),
        ("n_weak_material", "취약 재료 개수"),
        ("material_group", "대표 재료 그룹 (RC > 시멘트 > 흙·돌 > 나무)"),
        ("structure_type", "기초_지붕_1층 바닥 조합"),
        ("floor_band", "층수 구간 (1F / 2F / 3F+)"),
        ("weak_age", "흙·돌 건물의 나이"),
    ],
    "③ 지역 정보 (+32개)": [
        ("g1/g2/g3_building_count", "지역별 건물 수"),
        ("g1/g2/g3_weak_ratio, rc_ratio", "지역별 취약 재료·RC 건물 비율"),
        ("g1/g2/g3_age_gap", "건물 나이 − 지역 평균 나이"),
        ("*_low_rate, *_high_rate (16개)", "지역·재료·구조별 1등급/3등급 비율 (5-fold 타깃 인코딩)"),
        ("g2/g3_material_high/low_gap", "지역×재료 피해율 − 같은 재료 전체 피해율"),
    ],
}

# 하이퍼파라미터 무작위 탐색 상위 (실험 ③, 검증)
TUNING = pd.DataFrame([
    ["시도 5", 0.7207, 0.7097, 526, "1.9 / 1.0 / 1.3", 0.03, 127, 60],
    ["시도 19", 0.7197, 0.7074, 465, "1.6 / 1.0 / 1.2", 0.08, 31, 60],
    ["시도 25", 0.7195, 0.7083, 488, "1.7 / 1.0 / 1.2", 0.08, 31, 60],
    ["시도 15", 0.7195, 0.7079, 715, "2.0 / 1.0 / 1.4", 0.05, 31, 60],
    ["시도 22", 0.7193, 0.7075, 330, "1.8 / 1.0 / 1.2", 0.05, 63, 60],
    ["튜닝 전", 0.7184, 0.7067, 386, "1.8 / 1.0 / 1.3", 0.04, 31, 30],
], columns=["설정", "배수 적용 후 F1", "배수 적용 전 F1", "트리 수", "배수 (1/2/3)", "learning_rate", "num_leaves", "cat_smooth"])

# Soft voting (class_weight 1.9 / 1.0 / 1.3)
SINGLE = pd.DataFrame([
    ["LightGBM", 0.7187, 0.7493, 0.7119, 0.7277],
    ["HGB", 0.7168, 0.7467, 0.7073, 0.7294],
    ["XGBoost", 0.7123, 0.7450, 0.7078, 0.7186],
    ["ExtraTrees", 0.7123, 0.7474, 0.7103, 0.7169],
], columns=["모델", "Macro F1", "Accuracy", "Precision", "Recall"])

COMBOS = pd.DataFrame([
    ["HGB + LightGBM", 2, 0.7195, 0.7495, 0.7118, 0.7296],
    ["ExtraTrees + HGB + LightGBM", 3, 0.7195, 0.7506, 0.7140, 0.7272],
    ["ExtraTrees + HGB + XGBoost", 3, 0.7187, 0.7501, 0.7136, 0.7260],
    ["HGB + XGBoost + LightGBM", 3, 0.7181, 0.7488, 0.7117, 0.7266],
    ["ExtraTrees + HGB", 2, 0.7180, 0.7494, 0.7121, 0.7265],
    ["ExtraTrees + LightGBM", 2, 0.7179, 0.7501, 0.7142, 0.7237],
    ["HGB + XGBoost", 2, 0.7171, 0.7477, 0.7104, 0.7261],
    ["XGBoost + LightGBM", 2, 0.7171, 0.7482, 0.7118, 0.7242],
    ["ExtraTrees + XGBoost + LightGBM", 3, 0.7169, 0.7489, 0.7131, 0.7227],
    ["ExtraTrees + XGBoost", 2, 0.7161, 0.7490, 0.7133, 0.7211],
], columns=["조합", "모델 수", "Macro F1", "Accuracy", "Precision", "Recall"])

# 단계별 성능 흐름 (검증 Macro F1)
JOURNEY = pd.DataFrame([
    ["베이스라인\nCatBoost + SMOTENC", 0.6671, "원본 38개 변수, 원핫 인코딩"],
    ["실험 ①\n기존 파생변수", 0.7182, "LightGBM, 지역 ID 범주형, 배수 조정"],
    ["실험 ②\n+ 추가 변수", 0.7178, "변수 61개"],
    ["실험 ③\n+ 지역 정보", 0.7184, "변수 93개"],
    ["튜닝\n(시도 5)", 0.7207, "무작위 탐색 30회"],
    ["Soft voting\nHGB + LightGBM", 0.7195, "class_weight, 배수 없음"],
], columns=["단계", "검증 Macro F1", "설명"])

# 테스트 결과 비교
TEST_COMPARE = pd.DataFrame([
    ["LightGBM 튜닝 전 + 배수", 0.7137, 0.7467, 0.7076, 0.7213, 0.6800, 0.7876, 0.6964],
    ["LightGBM 튜닝 후 + 배수", 0.7155, 0.7476, 0.7087, 0.7240, 0.6859, 0.7858, 0.7004],
    ["Soft voting HGB + LightGBM", 0.7158, 0.7473, 0.7080, 0.7259, 0.6949, 0.7857, 0.6970],
], columns=["방식", "Macro F1", "Accuracy", "Precision", "Recall", "1등급 Recall", "2등급 Recall", "3등급 Recall"])

FINAL_CM = [[2619, 1086, 64], [1523, 17474, 3242], [161, 3803, 9119]]

# 학습 데이터 양에 따른 성능 (HGB + LightGBM)
SIZE_CURVE = pd.DataFrame([
    [0.10, 18242, 0.9590, 0.6949], [0.25, 45605, 0.8684, 0.7051], [0.50, 91210, 0.8101, 0.7130],
    [0.75, 136815, 0.7831, 0.7174], [1.00, 182420, 0.7710, 0.7195],
], columns=["비율", "학습 건수", "학습 F1", "검증 F1"])

# 변수 중요도 상위 20
GAIN = pd.Series({
    "g3_material_low_rate": 18.40, "g3_material_high_rate": 18.08, "g2_material_low_rate": 11.05,
    "g2_material_high_rate": 5.97, "g3_material_high_gap": 5.20, "geo_level_2_id": 5.10,
    "geo_level_3_id_high_rate": 4.61, "geo_level_3_id_low_rate": 2.52, "g3_age_gap": 1.96,
    "geo_level_3_id": 1.89, "g3_material_low_gap": 1.81, "geo_level_2_id_high_rate": 1.47,
    "g2_age_gap": 1.30, "g3_building_count": 1.08, "structure_type_low_rate": 1.08,
    "structure_type_high_rate": 0.96, "g1_age_gap": 0.94, "g2_material_high_gap": 0.93,
    "g1_material_low_rate": 0.86, "geo_level_2_id_low_rate": 0.83,
})
PERMUTATION = pd.DataFrame([
    ["g3_material_low_rate", 0.0477, 0.0021], ["g3_material_high_rate", 0.0283, 0.0030],
    ["geo_level_3_id_low_rate", 0.0267, 0.0009], ["g2_material_low_rate", 0.0250, 0.0040],
    ["g3_age_gap", 0.0064, 0.0006], ["geo_level_1_id", 0.0054, 0.0008],
    ["g2_material_high_rate", 0.0054, 0.0002], ["count_families", 0.0052, 0.0007],
    ["g3_building_count", 0.0041, 0.0011], ["structure_type", 0.0040, 0.0004],
    ["geo_level_3_id_high_rate", 0.0038, 0.0014], ["geo_level_2_id_high_rate", 0.0037, 0.0008],
    ["geo_level_2_id_low_rate", 0.0032, 0.0003], ["area_percentage", 0.0027, 0.0008],
    ["geo_level_2_id", 0.0026, 0.0007], ["age", 0.0024, 0.0006],
    ["material_count", 0.0024, 0.0020], ["geo_level_3_id", 0.0023, 0.0009],
    ["surface_condition", 0.0021, 0.0008], ["g2_age_gap", 0.0017, 0.0004],
], columns=["변수", "F1 감소", "표준편차"])
SHAP_MEAN = pd.DataFrame([
    ["g2_material_low_rate", 0.5198, 0.0149, 0.1239], ["g3_material_high_rate", 0.0877, 0.1827, 0.3822],
    ["geo_level_2_id", 0.3985, 0.0970, 0.1264], ["g3_material_low_rate", 0.4201, 0.0670, 0.1159],
    ["g2_material_high_rate", 0.0845, 0.0654, 0.1526], ["geo_level_3_id", 0.2343, 0.0271, 0.0326],
    ["geo_level_3_id_high_rate", 0.0226, 0.0683, 0.1514], ["geo_level_3_id_low_rate", 0.1843, 0.0421, 0.0145],
    ["g3_age_gap", 0.1313, 0.0192, 0.0820], ["ground_other_floor", 0.0960, 0.0450, 0.0528],
    ["g3_material_high_gap", 0.0120, 0.1484, 0.0239], ["structure_type", 0.0752, 0.0190, 0.0578],
    ["g2_age_gap", 0.0909, 0.0093, 0.0337], ["g1_material_high_rate", 0.0523, 0.0034, 0.0741],
    ["geo_level_1_id", 0.0506, 0.0220, 0.0559], ["g1_material_low_rate", 0.1022, 0.0061, 0.0161],
    ["geo_level_2_id_high_rate", 0.0252, 0.0374, 0.0568], ["g3_building_count", 0.0692, 0.0191, 0.0213],
    ["area_percentage", 0.0118, 0.0440, 0.0499], ["structure_type_low_rate", 0.0401, 0.0059, 0.0496],
], columns=["변수", "1등급", "2등급", "3등급"])
