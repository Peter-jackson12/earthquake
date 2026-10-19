"""네팔 건물 피해 데이터 전처리 v1 — 원본 보존, 고정 분할, fold 내부 인코딩.

Python 3.11+ recommended. Run from the extracted project directory:
    python preprocess.py --input ../nepal_building_damage.csv --output data

This script preprocesses and validates data. It never trains a predictive model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import PredefinedSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ID = 'building_id'
TARGET = 'damage_grade'
GEO = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
NUMERIC = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
TEXT = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
        'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
SUPERSTRUCTURE = [
    'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
    'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
    'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
    'has_superstructure_timber', 'has_superstructure_bamboo',
    'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
    'has_superstructure_other',
]
SECONDARY_DETAIL = [
    'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental',
    'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry',
    'has_secondary_use_health_post', 'has_secondary_use_gov_office',
    'has_secondary_use_use_police', 'has_secondary_use_other',
]
BINARY = SUPERSTRUCTURE + ['has_secondary_use'] + SECONDARY_DETAIL
CATEGORICAL = GEO + TEXT
FEATURES = (GEO + NUMERIC[:4] + TEXT[:7] + SUPERSTRUCTURE
            + ['legal_ownership_status', 'count_families', 'has_secondary_use'] + SECONDARY_DETAIL)
SOURCE_COLUMNS = [ID] + FEATURES + [TARGET]
ALLOWED_TEXT = {
    'land_surface_condition': list('not'), 'foundation_type': list('hiruw'),
    'roof_type': list('nqx'), 'ground_floor_type': list('fmvxz'),
    'other_floor_type': list('jqsx'), 'position': list('jost'),
    'plan_configuration': list('acdfmnoqsu'), 'legal_ownership_status': list('arvw'),
}
EXPECTED_SOURCE_SHA256 = '87abd2bc2b05d3e687bd176959967fd38334fff607e6afeb2d20d0d899c13e97'


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _integer_series(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors='raise')
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f'{name}: missing, infinite, or non-integer values found.')
    if (array < 0).any() or (array > np.iinfo(np.int32).max).any():
        raise ValueError(f'{name}: negative value or value outside the supported int32 range.')
    return values.astype('int32')


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """통계 학습 없는 형 정규화. ID/타깃/알 수 없는 추가 열은 실수 방지를 위해 거절."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Features must be a pandas DataFrame with named columns.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate feature column names found.')
    missing, extra = set(FEATURES) - set(frame.columns), set(frame.columns) - set(FEATURES)
    if missing or extra:
        raise ValueError(f'Feature schema mismatch. Missing={sorted(missing)}; extra={sorted(extra)}')
    if frame.isna().any().any():
        raise ValueError('Missing feature values found; v1 does not silently impute.')
    out = frame.loc[:, FEATURES].copy()
    for col in GEO + NUMERIC + BINARY:
        out[col] = _integer_series(out[col], col)
    for col in BINARY:
        if not out[col].isin([0, 1]).all():
            raise ValueError(f'{col}: expected only 0/1.')
        out[col] = out[col].astype('uint8')
    if (out['count_floors_pre_eq'] < 1).any():
        raise ValueError('count_floors_pre_eq must be at least 1.')
    for col in ['area_percentage', 'height_percentage']:
        if (out[col] > 100).any():
            raise ValueError(f'{col}: normalized percentage exceeds 100.')
    for col in GEO:
        # Geographic IDs are nominal identifiers, not continuous distances.
        out[col] = out[col].astype(str)
    for col in TEXT:
        out[col] = out[col].astype(str).str.strip()
        if out[col].eq('').any():
            raise ValueError(f'{col}: blank category found.')
        # Novel nonblank categories are allowed at transform time and handled by OHE.
    return out


def load_source(path: Path | str, verify_source_hash: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Data file not found: {path}')
    if verify_source_hash and sha256_file(path) != EXPECTED_SOURCE_SHA256:
        raise ValueError('Input SHA-256 differs from the approved source. Do not silently replace the dataset.')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle))
    if header != SOURCE_COLUMNS:
        raise ValueError('Source header or column order differs from the 40-column source contract.')
    raw = pd.read_csv(path, keep_default_na=False, encoding='utf-8-sig')
    out = normalize_features(raw[FEATURES])
    out.insert(0, ID, _integer_series(raw[ID], ID))
    out[TARGET] = _integer_series(raw[TARGET], TARGET)
    if out[ID].duplicated().any() or out[ID].eq(0).any():
        raise ValueError('building_id must be positive and unique.')
    if not out[TARGET].isin([1, 2, 3]).all():
        raise ValueError('damage_grade must be 1, 2, or 3.')
    out[TARGET] = out[TARGET].astype('uint8')
    for col, accepted in ALLOWED_TEXT.items():
        if not out[col].isin(accepted).all():
            raise ValueError(f'{col}: source contains a category outside the dictionary contract.')
    return out


class FeatureSchema(BaseEstimator, TransformerMixin):
    """학습 통계를 저장하지 않는 변수 스키마 검사기."""
    def fit(self, X: pd.DataFrame, y: Any = None) -> 'FeatureSchema':
        normalized = normalize_features(X)
        self.n_features_in_ = normalized.shape[1]
        self.feature_names_in_ = np.asarray(normalized.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return normalize_features(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURES, dtype=object)


def _float32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def make_preprocessor(scale_numeric: bool = False) -> Pipeline:
    """새 미학습 전처리기. 반드시 각 CV train fold에서 별도로 fit할 것."""
    cast = FunctionTransformer(_float32, feature_names_out='one-to-one')
    numeric_steps = [('float32', cast)]
    if scale_numeric:
        numeric_steps.append(('scale', StandardScaler()))
    columns = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), NUMERIC),
        ('binary', FunctionTransformer(_float32, feature_names_out='one-to-one'), BINARY),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32), CATEGORICAL),
    ], remainder='drop', sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([('schema', FeatureSchema()), ('columns', columns)])


def make_model_pipeline(estimator: Any, scale_numeric: bool = False) -> Pipeline:
    """예측 모델 학습은 하지 않고, 새 전처리기와 모델을 결합해 반환."""
    return Pipeline([('preprocess', make_preprocessor(scale_numeric)), ('model', estimator)])


def make_splits(data: pd.DataFrame, seed: int = 42, holdout_fraction: float = 0.2,
                n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data[ID].duplicated().any():
        raise ValueError('Cannot split duplicate building IDs.')
    # Freeze an explicit canonical order so source row reordering cannot change memberships.
    ordered = data.sort_values(ID, kind='stable').reset_index(drop=True)
    dev_pos, hold_pos = train_test_split(np.arange(len(ordered)), test_size=holdout_fraction,
                                       stratify=ordered[TARGET], random_state=seed)
    dev = ordered.iloc[np.sort(dev_pos)].reset_index(drop=True)
    hold = ordered.iloc[np.sort(hold_pos)].reset_index(drop=True)
    fold_ids = np.full(len(dev), -1, dtype=np.int8)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_pos) in enumerate(cv.split(dev[FEATURES], dev[TARGET])):
        fold_ids[valid_pos] = fold
    dev_manifest = pd.DataFrame({ID: dev[ID], 'split': 'development', 'cv_fold': fold_ids})
    hold_manifest = pd.DataFrame({ID: hold[ID], 'split': 'final_holdout', 'cv_fold': -1})
    manifest = pd.concat([dev_manifest, hold_manifest], ignore_index=True).sort_values(ID).reset_index(drop=True)
    return dev, hold, manifest


def load_development(output_dir: Path | str = 'data') -> tuple[pd.DataFrame, pd.Series, pd.Series, PredefinedSplit]:
    """학습 입력 38열, 정답, 건물 ID, 저장된 CV를 정렬/ID 대조 후 반환. Holdout은 읽지 않음."""
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / 'development.csv', dtype={c: str for c in CATEGORICAL})
    if list(frame.columns) != SOURCE_COLUMNS or frame[ID].duplicated().any():
        raise ValueError('Invalid development table schema or duplicate ID.')
    manifest = pd.read_csv(output_dir / 'split_manifest.csv')
    if manifest[ID].duplicated().any():
        raise ValueError('Duplicate IDs in split manifest.')
    dev_manifest = manifest.loc[manifest['split'].eq('development')].set_index(ID)
    if set(frame[ID]) != set(dev_manifest.index):
        raise ValueError('Development IDs do not match the split manifest.')
    # Validate before casting: an integer cast alone silently truncates 0.5 to 0.
    # Only development rows are checked here; final_holdout keeps its -1 sentinel.
    fold_values = _integer_series(dev_manifest.loc[frame[ID], 'cv_fold'], 'cv_fold')
    folds = fold_values.to_numpy(dtype=int)
    if set(folds) != set(range(5)):
        raise ValueError('Expected the frozen five development folds, numbered 0..4.')
    labels = _integer_series(frame[TARGET], TARGET)
    if not labels.isin([1, 2, 3]).all():
        raise ValueError('Invalid development labels.')
    labels = labels.astype('uint8')
    X = normalize_features(frame[FEATURES])
    return X, labels, frame[ID].copy(), PredefinedSplit(folds)


def quality_audit(data: pd.DataFrame) -> dict[str, Any]:
    repeated = int(data.duplicated(subset=FEATURES).sum())
    conflicts = data.groupby(FEATURES, dropna=False, observed=True)[TARGET].nunique()
    return {
        'rows': len(data), 'columns': data.shape[1], 'input_features': len(FEATURES),
        'missing_cells': int(data.isna().sum().sum()), 'duplicate_ids': int(data[ID].duplicated().sum()),
        'duplicate_full_rows': int(data.duplicated().sum()), 'deleted_rows': 0, 'imputed_cells': 0,
        'feature_duplicates_beyond_first': repeated,
        'feature_groups_with_multiple_labels': int((conflicts > 1).sum()),
        'age_995_rows': int(data['age'].eq(995).sum()), 'age_0_rows': int(data['age'].eq(0).sum()),
        'zero_families_rows': int(data['count_families'].eq(0).sum()),
        'invalid_binary_cells': int((~data[BINARY].isin([0, 1])).sum().sum()),
        'geo3_multiple_geo2_parents': int((data.groupby(GEO[2])[GEO[1]].nunique() > 1).sum()),
        'geo2_multiple_geo1_parents': int((data.groupby(GEO[1])[GEO[0]].nunique() > 1).sum()),
        'secondary_use_contradictions': int((data['has_secondary_use'] != data[SECONDARY_DETAIL].max(axis=1)).sum()),
        'target_counts': {str(k): int(v) for k, v in data[TARGET].value_counts().sort_index().items()},
        'geographic_cardinality': {c: int(data[c].nunique()) for c in GEO},
        'predictive_model_trained': False,
        'holdout_policy': 'Only source integrity checks, stratification and file separation. Not used to fit preprocessing, select models or compute model scores.',
    }


def _matrix_stats(matrix: Any) -> dict[str, Any]:
    matrix = sparse.csr_matrix(matrix)
    if matrix.dtype != np.float32 or not np.isfinite(matrix.data).all():
        raise AssertionError('Expected finite float32 sparse matrix.')
    return {'rows': matrix.shape[0], 'columns': matrix.shape[1], 'nnz': matrix.nnz,
            'dtype': str(matrix.dtype), 'format': matrix.format,
            'storage_mib': round((matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 2**20, 3),
            'dense_float32_gib_estimate': round(matrix.shape[0] * matrix.shape[1] * 4 / 2**30, 3)}


def validate_fold_preprocessing(output_dir: Path | str) -> list[dict[str, Any]]:
    """5개 fold 전체의 실제 fit/transform 검증. Holdout과 예측 모델에는 접근하지 않음."""
    X, y, ids, cv = load_development(output_dir)
    results = []
    for fold, (train_pos, valid_pos) in enumerate(cv.split()):
        train, valid = X.iloc[train_pos], X.iloc[valid_pos]
        encoder = make_preprocessor()
        encoded_train = encoder.fit_transform(train)
        encoded_valid = encoder.transform(valid)
        if not sparse.issparse(encoded_train) or not sparse.issparse(encoded_valid):
            raise AssertionError('Sparse output expected; do not materialize a dense matrix.')
        if encoded_train.shape[1] != encoded_valid.shape[1]:
            raise AssertionError('Train/validation encoded column mismatch.')
        names = encoder.get_feature_names_out().tolist()
        if len(names) != encoded_train.shape[1] or len(set(names)) != len(names):
            raise AssertionError('Invalid encoded feature names.')
        if any(ID in name or TARGET in name for name in names):
            raise AssertionError('Identifier or target leaked into preprocessing.')
        ohe = encoder.named_steps['columns'].named_transformers_['categorical']
        novel = {}
        for col, learned in zip(CATEGORICAL, ohe.categories_):
            if set(learned) != set(train[col].unique()):
                raise AssertionError(f'{col}: vocabulary was not fitted only on the train fold.')
            n_unknown = int((~valid[col].isin(learned)).sum())
            if n_unknown:
                novel[col] = {'rows': n_unknown, 'categories': int(valid.loc[~valid[col].isin(learned), col].nunique())}
        item = {'fold': fold, 'train': _matrix_stats(encoded_train), 'validation': _matrix_stats(encoded_valid),
                'unseen_validation_categories': novel, 'train_only_vocabulary_checked': True,
                'feature_name_count': len(names), 'status': 'PASS'}
        results.append(item)
        print(f"Fold {fold}: {len(train):,} train / {len(valid):,} validation; {len(names):,} encoded columns; PASS", flush=True)
    return results


def prepare(input_path: Path | str, output_dir: Path | str = 'data', *, overwrite: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'{output_dir} is not empty. Use a new directory or explicitly pass --overwrite.')
    source_hash = sha256_file(input_path)
    data = load_source(input_path)
    audit = quality_audit(data)
    dev, hold, manifest = make_splits(data)
    if set(dev[ID]) & set(hold[ID]) or len(dev) + len(hold) != len(data):
        raise AssertionError('Invalid development/holdout split.')
    # Determinism check includes a reversed input order; canonical ID ordering must give the same split.
    _, _, repeat = make_splits(data.iloc[::-1])
    pd.testing.assert_frame_equal(manifest, repeat)
    for directory in [output_dir, output_dir / 'final_holdout', output_dir / 'qa']:
        directory.mkdir(parents=True, exist_ok=True)
    dev.to_csv(output_dir / 'development.csv', index=False, lineterminator='\n')
    hold[[ID] + FEATURES].to_csv(output_dir / 'final_holdout' / 'features.csv', index=False, lineterminator='\n')
    hold[[ID, TARGET]].to_csv(output_dir / 'final_holdout' / 'labels_DO_NOT_USE_FOR_TUNING.csv', index=False, lineterminator='\n')
    manifest.to_csv(output_dir / 'split_manifest.csv', index=False, lineterminator='\n')
    (output_dir / 'final_holdout' / 'README.txt').write_text(
        '최종 평가 전까지 사용하지 마세요.\n이 폴더는 기술적으로 잠기거나 암호화된 상태가 아닙니다.\n'
        'features.csv와 labels_DO_NOT_USE_FOR_TUNING.csv는 building_id로 결합합니다.\n'
        '공식 대회 test 파일이 아니라, 제공된 정답 포함 데이터에서 별도로 남긴 내부 holdout입니다.\n', encoding='utf-8')
    distribution_rows = []
    for name, subset in [('all', data), ('development', dev), ('final_holdout', hold)]:
        for label, count in subset[TARGET].value_counts().sort_index().items():
            distribution_rows.append({'split': name, 'damage_grade': int(label), 'rows': int(count), 'share': float(count / len(subset))})
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(output_dir / 'qa' / 'class_distribution.csv', index=False, lineterminator='\n')
    dev_folds = manifest.loc[manifest['split'].eq('development')].set_index(ID)['cv_fold']
    fold_table = pd.crosstab(dev[ID].map(dev_folds), dev[TARGET]).rename_axis(index='fold', columns=TARGET)
    fold_table.to_csv(output_dir / 'qa' / 'fold_class_counts.csv', lineterminator='\n')
    # EDA-oriented profile is development-only; no holdout feature/target relationship exploration.
    profile = []
    for col in FEATURES:
        role = 'numeric' if col in NUMERIC else 'binary' if col in BINARY else 'geographic_category' if col in GEO else 'text_category'
        profile.append({'column': col, 'group': role, 'runtime_dtype': str(dev[col].dtype),
                        'development_unique_values': int(dev[col].nunique()), 'development_missing': int(dev[col].isna().sum()),
                        'development_min': int(dev[col].min()) if col in NUMERIC + BINARY else '',
                        'development_max': int(dev[col].max()) if col in NUMERIC + BINARY else ''})
    pd.DataFrame(profile).to_csv(output_dir / 'qa' / 'development_feature_profile.csv', index=False, lineterminator='\n')
    schema = {'schema_version': '1.0', 'id': ID, 'target': TARGET, 'features': FEATURES, 'numeric': NUMERIC,
              'binary': BINARY, 'geographic': GEO, 'text_categories': TEXT, 'categorical': CATEGORICAL,
              'source_columns': SOURCE_COLUMNS, 'labels': [1, 2, 3],
              'csv_note': 'CSV does not preserve dtypes. Use load_development()/normalize_features() to restore geographic IDs as strings.',
              'preprocessing': {'one_hot': 'fit separately inside each train fold', 'handle_unknown': 'ignore',
                                'sparse': True, 'output_dtype': 'float32', 'scaling_default': False,
                                'imputation': None, 'resampling': None, 'feature_engineering': None}}
    write_json(output_dir / 'schema.json', schema)
    audit.update({'development_rows': len(dev), 'holdout_rows': len(hold), 'source_sha256': source_hash,
                  'split_seed': 42, 'n_folds': 5, 'sort_before_split': ID,
                  'split_id_overlap': 0, 'reordered_source_split_reproducible': True,
                  'development_csv_value_roundtrip_checked': True})
    reread = pd.read_csv(output_dir / 'development.csv', dtype={c: str for c in CATEGORICAL})
    pd.testing.assert_frame_equal(normalize_features(reread[FEATURES]), dev[FEATURES], check_dtype=True)
    if reread[ID].tolist() != dev[ID].tolist() or reread[TARGET].tolist() != dev[TARGET].tolist():
        raise AssertionError('CSV roundtrip changed identifiers or labels.')
    fold_results = validate_fold_preprocessing(output_dir)
    write_json(output_dir / 'qa' / 'fold_preprocessing_validation.json', fold_results)
    audit['fold_preprocessing_validation'] = '5/5 PASS'
    write_json(output_dir / 'qa' / 'audit.json', audit)
    environment = {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
                   'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__}
    write_json(output_dir / 'qa' / 'environment.json', environment)
    if sha256_file(input_path) != source_hash:
        raise AssertionError('Source file changed during preprocessing.')
    # Byte-level hashes make the shipped split and dataset files directly auditable.
    hashes = {str(p.relative_to(output_dir)).replace('\\', '/'): sha256_file(p)
              for p in sorted(output_dir.rglob('*')) if p.is_file() and p.name != 'checksums.json'}
    write_json(output_dir / 'checksums.json', hashes)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description='Nepal preprocessing only; no predictive model training.')
    parser.add_argument('--input', type=Path, required=True, help='Original nepal_building_damage.csv (SHA-256 checked).')
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow rebuilding the output files.')
    args = parser.parse_args()
    try:
        audit = prepare(args.input, args.output, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError, AssertionError) as exc:
        parser.exit(1, f'Preprocessing failed: {exc}\n')
    print(json.dumps({key: audit[key] for key in ['rows', 'input_features', 'development_rows', 'holdout_rows', 'fold_preprocessing_validation', 'predictive_model_trained']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
