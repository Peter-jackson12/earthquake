"""Safety-only SMOTENC pilot on frozen fold-0 TRAIN rows, never a model score.

Original schema is checked BEFORE synthesis. Numeric columns are deliberately
float64, preventing silent dtype-induced rounding of synthetic values. The old
preprocessor is unchanged. Unsafe synthetic rows are counted, not repaired.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import GEO, NUMERIC, BINARY, CATEGORICAL, FEATURES, normalize_features, sha256_file, write_json
from scripts.audit_eda_oof import development_frame
CATS = CATEGORICAL + BINARY


def synthesize(X: pd.DataFrame, y: pd.Series, seed: int = 42, k: int = 5):
    source = normalize_features(X).reset_index(drop=True)
    target = y.reset_index(drop=True)
    if min(target.value_counts()) <= k:
        raise ValueError('Each class must contain more than k training samples.')
    # Fit only the pilot training population. No validation inputs accepted here.
    scaler = StandardScaler().fit(source[NUMERIC].astype('float64'))
    scaled = source.copy()
    for col in NUMERIC:
        scaled[col] = scaled[col].astype('float64')
    scaled.loc[:, NUMERIC] = scaler.transform(source[NUMERIC].astype('float64'))
    for col in CATS:
        scaled[col] = source[col].astype(str)
    sampler = SMOTENC(categorical_features=CATS, random_state=seed, k_neighbors=k)
    with threadpool_limits(limits=1):
        generated, labels = sampler.fit_resample(scaled, target)
    generated.loc[:, NUMERIC] = scaler.inverse_transform(generated[NUMERIC].astype(float))
    return source, generated, np.asarray(labels), scaler


def inspect_synthetic(original: pd.DataFrame, synthetic: pd.DataFrame, train_reference: pd.DataFrame):
    """Finite/binary checks and empirical joint-category consistency, not engineering certification."""
    counts = {}
    if len(synthetic) == 0:
        raise ValueError('No synthetic rows to inspect.')
    num = synthetic[NUMERIC].to_numpy(dtype=float)
    counts['nonfinite_numeric_rows'] = int((~np.isfinite(num).all(axis=1)).sum())
    counts['fractional_numeric_rows'] = int((~np.isclose(num, np.round(num), atol=1e-8, rtol=0)).any(axis=1).sum())
    for i,col in enumerate(NUMERIC):
        counts['fractional_' + col] = int((~np.isclose(num[:,i], np.round(num[:,i]), atol=1e-8, rtol=0)).sum())
    counts['out_of_training_numeric_range_rows'] = int(((num < original[NUMERIC].min().to_numpy()-1e-8) |
                                                        (num > original[NUMERIC].max().to_numpy()+1e-8)).any(axis=1).sum())
    cat_bad = np.zeros(len(synthetic), dtype=bool)
    for col in CATS:
        cat_bad |= ~synthetic[col].astype(str).isin(original[col].astype(str).unique()).to_numpy()
    counts['unseen_marginal_category_rows'] = int(cat_bad.sum())
    binary = synthetic[BINARY].apply(pd.to_numeric).to_numpy(dtype=float)
    counts['invalid_binary_rows'] = int((~np.isin(binary, [0.,1.])).any(axis=1).sum())
    known_triples = pd.MultiIndex.from_frame(train_reference[GEO].astype(str).drop_duplicates())
    new_triples = pd.MultiIndex.from_frame(synthetic[GEO].astype(str))
    counts['unobserved_geo_triple_rows'] = int((~new_triples.isin(known_triples)).sum())
    for child,parent in [('geo_level_2_id','geo_level_1_id'),('geo_level_3_id','geo_level_2_id')]:
        pairs = train_reference[[child,parent]].astype(str).drop_duplicates()
        unique = pairs.groupby(child)[parent].nunique()
        stable = pairs.loc[pairs[child].isin(unique[unique.eq(1)].index)].set_index(child)[parent]
        expected = synthetic[child].astype(str).map(stable)
        counts[f'{child}_parent_conflict_rows'] = int((expected.notna() & expected.ne(synthetic[parent].astype(str))).sum())
        counts[f'{child}_ambiguous_parent_ids_in_training'] = int(unique.gt(1).sum())
    regular = train_reference.loc[train_reference.age.ne(995), 'age']
    age = synthetic.age.to_numpy(dtype=float)
    counts['age_between_regular_max_and_995_rows'] = int(((age > regular.max()+1e-8) & (age < 995-1e-8)).sum())
    counts['reference_max_age_excluding_995'] = float(regular.max())
    reasons = []
    for key in ['nonfinite_numeric_rows', 'out_of_training_numeric_range_rows', 'unseen_marginal_category_rows', 'invalid_binary_rows']:
        if counts[key]:
            reasons.append(f'Failed synthetic domain check: {key}')
    if counts['fractional_numeric_rows']:
        reasons.append('Synthetic continuous values violate the unchanged original integer schema; no rounding policy approved.')
    if counts['unobserved_geo_triple_rows']:
        reasons.append('Individual valid region codes recombine into geographic tuples unobserved in fold training data.')
    if counts['age_between_regular_max_and_995_rows']:
        reasons.append('Unresolved age=995 code participates in interpolation beyond observed ordinary-age range.')
    return {'synthetic_rows':len(synthetic), 'counts':counts,
            'full_experiment_allowed': not reasons, 'blocking_reasons':reasons}


def peak_rss_mib():
    try:
        import resource
    except ImportError:  # Windows: do not invent a peak memory measurement.
        return None
    divisor = 1024**2 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def run(data: Path, output: Path, rows: int = 6000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; no silent overwrite.')
    if not 300 <= rows <= 10000:
        raise ValueError('Pilot budget requires 300..10000 rows.')
    frame, contract = development_frame(data)
    train = frame.loc[frame.fold.ne(0)].reset_index(drop=True)
    positions,_ = train_test_split(np.arange(len(train)),train_size=rows,
                                    stratify=train.damage_grade,random_state=42)
    positions=np.sort(positions)
    sample = train.iloc[positions].reset_index(drop=True)
    assert not set(sample.building_id) & set(frame.loc[frame.fold.eq(0),'building_id'])
    start=time.perf_counter()
    original, generated, labels, scaler = synthesize(sample[FEATURES],sample.damage_grade)
    elapsed=time.perf_counter()-start
    synthetic=generated.iloc[len(original):].reset_index(drop=True)
    report=inspect_synthetic(original,synthetic,train)
    # Original rows must be preserved before any claim about oversampling counts.
    np.testing.assert_allclose(generated.iloc[:len(original)][NUMERIC].to_numpy(dtype=float),
                               original[NUMERIC].to_numpy(dtype=float), atol=1e-8,rtol=0)
    for col in CATS:
        assert np.array_equal(generated.iloc[:len(original)][col].astype(str),original[col].astype(str))
    report.update(scope='fold0_training_only_safety_pilot_no_model_fit',
                  status='BLOCKED_FULL_EXPERIMENT' if report['blocking_reasons'] else 'PILOT_PASS_REVIEW_REQUIRED',
                  data_contract=contract, train_rows_available=len(train), pilot_rows=rows,
                  fold=0, k_neighbors=5, seed=42,
                  class_counts_before={str(k):int(v) for k,v in sample.damage_grade.value_counts().sort_index().items()},
                  class_counts_after={str(k):int(v) for k,v in pd.Series(labels).value_counts().sort_index().items()},
                  sampler_seconds=elapsed, process_peak_rss_mib=peak_rss_mib(),
                  imbalanced_learn_version=imblearn.__version__, numeric_columns=NUMERIC,
                  categorical_columns=CATS, original_schema_unchanged=True, validation_resampled=False,
                  holdout_opened=False, model_fitted=False, code_sha256=sha256_file(__file__),
                  input_ids_sha256=hashlib.sha256(sample.building_id.to_numpy(dtype='<i8').tobytes()).hexdigest(),
                  limitations=['No predictive metric is estimated by this safety-only pilot.',
                               'Observed joint-category consistency is not proof of engineering validity.',
                               'Process peak RSS is not incremental sampler memory; another fit may run concurrently.'])
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'smotenc_pilot.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/smotenc_safety_v1')
    p.add_argument('--rows',type=int,default=6000)
    a=p.parse_args();run(a.data,a.output,a.rows)
