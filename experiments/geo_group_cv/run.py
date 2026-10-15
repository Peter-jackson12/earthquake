"""Frozen geographic CV; existing C/G2 pipelines, no holdout or tuning.

prepare writes both manifests and a pre-fit design seal. fit requires that seal.
Outputs are private; reports are separately reviewed aggregate exports.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from preprocess import FEATURES, CATEGORICAL, NUMERIC, ID, load_development, sha256_file
from experiments.age995.run import make_pipeline as make_C, PARAMS
from experiments.age995.interactions.run import make_pipeline as make_G2, raw_terms

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PROBS = ['p_grade_1', 'p_grade_2', 'p_grade_3']
SOURCE = ['preprocess.py', 'baseline.py', 'scripts/audit_eda_oof.py',
          'experiments/age995/run.py', 'experiments/age995/audit_followup.py',
          'experiments/age995/interactions/run.py', 'experiments/age995/interactions/config.json',
          'experiments/geo_group_cv/run.py', 'experiments/geo_group_cv/config.json']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def cfg():
    c = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
    require(c['lr'] == PARAMS and c['variants'] == ['C', 'G2'], 'Fixed model policy changed.')
    require(c['n_splits'] == 5 and c['seed'] == 42 and c['shuffle'], 'Fixed split policy changed.')
    require(sklearn.__version__ == c['sklearn_version'], 'Splitter version must be 1.8.0.')
    return c


def source_hashes():
    return {p: sha256_file(ROOT/p) for p in SOURCE}


def load(data):
    contract = json.loads((ROOT/'configs/data_contract.json').read_text())
    for name, digest in contract['files'].items():
        require(sha256_file(Path(data)/name) == digest, 'Data SHA mismatch: '+name)
    X, y, ids, old_cv = load_development(data)
    require(len(X) == contract['expected_rows'] and list(X.columns) == FEATURES, 'Data dimensions changed.')
    order = np.argsort(ids.to_numpy(), kind='stable')
    return (X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True),
            ids.iloc[order].reset_index(drop=True), old_cv.test_fold[order], contract)


def groups(X, columns):
    values = X[columns].astype('int64').to_numpy()
    unique, codes = np.unique(values, axis=0, return_inverse=True)
    return codes, unique


def validate_assignment(X, y, ids, folds, columns, c):
    require(len(X) == len(y) == len(ids) == len(folds), 'Unaligned split.')
    require(pd.Series(ids).is_unique, 'Duplicate IDs.')
    folds = np.asarray(folds)
    require(np.isfinite(folds).all() and (folds == np.floor(folds)).all(), 'Invalid fold values.')
    require(set(folds) == set(range(5)), 'Expected five nonempty folds.')
    g, keys = groups(X, columns)
    require(len(keys) >= 5, 'Fewer than five geographic groups.')
    table = pd.DataFrame({'group': g, 'fold': folds})
    require(table.groupby('group').fold.nunique().eq(1).all(), 'Group leakage.')
    base = np.bincount(np.asarray(y, dtype=int), minlength=4)[1:4]/len(y)
    rows = []
    for f in range(5):
        a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        require(len(a) > 0 and len(b) > 0, 'Empty train/validation.')
        require(not set(g[a]) & set(g[b]), 'Train/validation group overlap.')
        require(not set(np.asarray(ids)[a]) & set(np.asarray(ids)[b]), 'Train/validation ID overlap.')
        ta = np.bincount(np.asarray(y)[a].astype(int), minlength=4)[1:4]
        va = np.bincount(np.asarray(y)[b].astype(int), minlength=4)[1:4]
        require((ta >= c['min_class_support']).all() and (va >= c['min_class_support']).all(), 'Class support gate failed.')
        require(c['min_validation_fraction'] <= len(b)/len(y) <= c['max_validation_fraction'], 'Validation size gate failed.')
        row = dict(fold=f, train_rows=len(a), valid_rows=len(b),
                   train_groups=len(set(g[a])), valid_groups=len(set(g[b])), group_overlap=0, id_overlap=0,
                   train_support=ta.tolist(), valid_support=va.tolist(), valid_class_fraction=(va/len(b)).tolist(),
                   max_class_fraction_deviation=float(np.max(abs(va/len(b)-base))))
        rows.append(row)
    ratio = max(r['valid_rows'] for r in rows)/min(r['valid_rows'] for r in rows)
    notes = []
    if ratio > c['warn_fold_size_ratio']:
        notes.append('Fold validation-size ratio exceeds 3; distribution-shift stress test only.')
    if max(r['max_class_fraction_deviation'] for r in rows) > c['warn_class_fraction_deviation']:
        notes.append('Class fraction deviation exceeds 0.10; distribution-shift stress test only.')
    return dict(status='PASS_WITH_LIMITATIONS' if notes else 'PASS', folds=rows,
                validation_size_ratio=ratio, warnings=notes, groups=len(keys))


def prepare(data, output):
    c = cfg()
    output = Path(output)
    require(not (output/'design.json').exists(), 'Do not overwrite an existing design.')
    X, y, ids, old, contract = load(data)
    design = dict(study_id=c['study_id'], config=c, data_contract=contract,
                  source_sha256=source_hashes(), created_at_utc=datetime.now(timezone.utc).isoformat(),
                  classifiers_fitted=0, holdout_accessed=False, schemes={})
    design['hierarchy'] = dict(geo1_groups=int(X.geo_level_1_id.nunique()),
        geo2_codes=int(X.geo_level_2_id.nunique()), geo3_codes=int(X.geo_level_3_id.nunique()),
        geo2_codes_with_multiple_geo1_parents=int(X.groupby('geo_level_2_id').geo_level_1_id.nunique().gt(1).sum()),
        geo3_codes_with_multiple_geo2_parents=int(X.groupby('geo_level_3_id').geo_level_2_id.nunique().gt(1).sum()))
    for name, columns in c['schemes'].items():
        g, keys = groups(X, columns)
        folds = np.full(len(X), -1, dtype=int)
        coverage = np.zeros(len(X), dtype=int)
        split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for f, (a, b) in enumerate(split.split(X, y, g)):
            folds[b] = f
            coverage[b] += 1
        require(np.all(coverage == 1), 'OOF coverage must equal one.')
        result = validate_assignment(X, y, ids, folds, columns, c)
        d = output/name
        d.mkdir(parents=True, exist_ok=False)
        manifest = pd.DataFrame({ID: ids, 'fold': folds, **{col: X[col] for col in columns}})
        manifest.to_csv(d/'manifest.csv', index=False, lineterminator='\n')
        sizes = pd.Series(g).value_counts()
        result.update(manifest_sha256=sha256_file(d/'manifest.csv'), columns=columns,
                      group_size=sizes.describe().to_dict())
        group_stats = X[columns].copy()
        group_stats['damage_grade'] = y
        group_stats['fold'] = folds
        summary = group_stats.groupby(columns+['fold','damage_grade'], observed=True).size().unstack(fill_value=0)
        summary.reindex(columns=[1,2,3], fill_value=0).to_csv(d/'group_support.csv')
        result['group_support_sha256'] = sha256_file(d/'group_support.csv')
        design['schemes'][name] = result
    write(output/'design.json', design)
    write(output/'design_seal.json', {'design_sha256':sha256_file(output/'design.json'),
          'created_at_utc':datetime.now(timezone.utc).isoformat(), 'classifiers_fitted':0})
    return design


def verify_design(data, output):
    output = Path(output)
    seal = json.loads((output/'design_seal.json').read_text())
    require(sha256_file(output/'design.json') == seal['design_sha256'], 'Design seal changed.')
    design = json.loads((output/'design.json').read_text())
    require(design['source_sha256'] == source_hashes() and design['config'] == cfg(), 'Frozen source changed.')
    X, y, ids, old, contract = load(data)
    require(design['data_contract'] == contract, 'Data contract changed.')
    manifests = {}
    for name, spec in design['schemes'].items():
        path = output/name/'manifest.csv'
        require(sha256_file(path) == spec['manifest_sha256'], 'Manifest SHA changed.')
        m = pd.read_csv(path, dtype={col:str for col in spec['columns']})
        np.testing.assert_array_equal(m[ID], ids)
        for col in spec['columns']:
            np.testing.assert_array_equal(m[col], X[col])
        validate_assignment(X, y, ids, m.fold.to_numpy(), spec['columns'], cfg())
        manifests[name] = m
    return X, y, ids, manifests, design


def score(y, pred):
    cm = confusion_matrix(y, pred, labels=[1,2,3])
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[1,2,3], zero_division=0)
    norm = np.divide(cm, s[:,None], out=np.zeros((3,3)), where=s[:,None] != 0)
    return dict(n=len(y), macro_f1=float(f.mean()), accuracy=float(accuracy_score(y,pred)),
                precision=p.tolist(), recall=r.tolist(), f1=f.tolist(), support=s.tolist(),
                confusion=cm.tolist(), normalized_confusion=norm.tolist(),
                error31=int(cm[2,0]), error32=int(cm[2,1]),
                rate31=float(cm[2,0]/s[2]) if s[2] else None,
                rate32=float(cm[2,1]/s[2]) if s[2] else None)


def preprocessing_evidence(pipe, variant, train, valid):
    base = pipe['preprocess'] if variant == 'C' else dict(pipe['preprocess'].transformer_list)['C']
    scaler = base['columns'].named_transformers_['numeric']['scale']
    normal = base['schema'].transform(train)
    age = base['age995']
    require(age.median_ == float(train.loc[train.age.ne(995),'age'].median()), 'Invalid fold median.')
    expected = base['age995'].transform(normal)[NUMERIC].to_numpy(dtype=np.float32).astype(np.float64)
    np.testing.assert_allclose(scaler.mean_, expected.mean(0), atol=1e-10, rtol=0)
    np.testing.assert_allclose(scaler.var_, expected.var(0), atol=1e-8, rtol=1e-10)
    require(int(scaler.n_samples_seen_) == len(train), 'Scaler denominator differs.')
    before = joblib.hash(pipe['preprocess'])
    enc = base['columns'].named_transformers_['categorical']
    cat_matrix = enc.transform(valid[CATEGORICAL])
    records = []
    offset = 0
    for col, learned in zip(CATEGORICAL, enc.categories_):
        require(set(learned) == set(train[col]), 'Nontraining category in encoder.')
        unknown = ~valid[col].isin(learned).to_numpy()
        require(cat_matrix[unknown, offset:offset+len(learned)].nnz == 0, 'Unknown category not encoded as zero.')
        records.append(dict(feature=col, train_categories=len(learned), valid_categories=int(valid[col].nunique()),
                            unknown_rows=int(unknown.sum()), unknown_unique=int(valid.loc[unknown,col].nunique()),
                            unknown_block_zero=True))
        offset += len(learned)
    extra_records = []
    if variant == 'G2':
        extra = dict(pipe['preprocess'].transformer_list)['extra']
        terms_train = raw_terms(train)[list(extra['terms'].columns_)]
        terms_valid = raw_terms(valid)[list(extra['terms'].columns_)]
        mat = extra['encode'].transform(terms_valid)
        offset = 0
        for col, learned in zip(terms_train, extra['encode'].categories_):
            require(set(learned) == set(terms_train[col]), 'Extra vocabulary contamination.')
            unknown = ~terms_valid[col].isin(learned).to_numpy()
            require(mat[unknown, offset:offset+len(learned)].nnz == 0, 'Unknown extra block not zero.')
            extra_records.append(dict(feature=col, train_categories=len(learned), unknown_rows=int(unknown.sum()),
                                      unknown_unique=int(terms_valid.loc[unknown,col].nunique()), unknown_block_zero=True))
            offset += len(learned)
    matrix = pipe['preprocess'].transform(valid)
    require(sparse.isspmatrix_csr(matrix) and matrix.dtype == np.float32, 'Expected float32 CSR.')
    require(joblib.hash(pipe['preprocess']) == before, 'Validation transform mutated preprocessing.')
    return dict(median=age.median_, numeric_mean=scaler.mean_.tolist(), numeric_var=scaler.var_.tolist(),
                numeric_scale=scaler.scale_.tolist(), n_samples_seen=int(scaler.n_samples_seen_),
                train_age995=int(train.age.eq(995).sum()), valid_age995=int(valid.age.eq(995).sum()),
                categorical=records, extra=extra_records, encoded_columns=matrix.shape[1],
                preprocessing_before_after_equal=True, input_columns=list(train.columns), csr_float32=True)


def fit(data, output, scheme, variant, code_commit):
    require(scheme in cfg()['schemes'] and variant in cfg()['variants'], 'Unplanned comparison.')
    require(len(code_commit) == 40 and all(c in '0123456789abcdef' for c in code_commit), 'Full source commit required.')
    X, y, ids, manifests, design = verify_design(data, output)
    directory = Path(output)/scheme/variant
    directory.mkdir(parents=True, exist_ok=False)
    meta = dict(status='running', scheme=scheme, variant=variant, code_commit=code_commit,
                design_sha256=sha256_file(Path(output)/'design.json'), source_sha256=source_hashes(),
                data_contract=design['data_contract'], started_at_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                 scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
                holdout_accessed=False, attempted_fits=0, completed_fits=0, threads=1)
    write(directory/'metadata.json', meta)
    folds = manifests[scheme].fold.to_numpy()
    prob = np.full((len(X),3), np.nan)
    pred = np.full(len(X), -1, dtype=int)
    seen = np.zeros(len(X), dtype=int)
    records = []
    data_before = pd.util.hash_pandas_object(X, index=True).to_numpy()
    try:
        for f in range(5):
            a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            pipe = make_C('C') if variant == 'C' else make_G2('G2')
            meta['attempted_fits'] += 1
            write(directory/'metadata.json', meta)
            start = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipe.fit(X.iloc[a], y.iloc[a])
            seconds = time.perf_counter()-start
            require(np.array_equal(pipe['model'].classes_, [1,2,3]), 'Unexpected model class order.')
            with threadpool_limits(limits=1):
                pred[b] = pipe.predict(X.iloc[b])
                prob[b] = pipe.predict_proba(X.iloc[b])
                check = preprocessing_evidence(pipe, variant, X.iloc[a], X.iloc[b])
                model_path = directory/f'fold_{f}.joblib'
                joblib.dump(pipe, model_path, compress=3)
                reloaded = joblib.load(model_path)  # Only our freshly generated trusted model.
                np.testing.assert_array_equal(reloaded.predict(X.iloc[b]), pred[b])
                np.testing.assert_array_equal(reloaded.predict_proba(X.iloc[b]), prob[b])
            seen[b] += 1
            record = dict(fold=f, train_rows=len(a), valid_rows=len(b), metrics=score(y.iloc[b],pred[b]),
                          fit_seconds=seconds, n_iter=pipe['model'].n_iter_.tolist(),
                          converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught),
                          warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                          preprocessing=check, model_sha256=sha256_file(model_path), reload_exact=True)
            records.append(record)
            write(directory/'folds.json',records)
            meta['completed_fits'] += 1
            write(directory/'metadata.json',meta)
            print(f'{scheme}/{variant} fold={f} F1={record["metrics"]["macro_f1"]:.9f} fit={seconds:.1f}s '
                  f'iter={record["n_iter"]} converged={record["converged"]}',flush=True)
        require(np.all(seen == 1), 'OOF coverage changed.')
        np.testing.assert_array_equal(data_before,pd.util.hash_pandas_object(X,index=True).to_numpy())
        oof = pd.DataFrame({ID:ids,'fold':folds,'y_true':y,'y_pred':pred, **{c:prob[:,i] for i,c in enumerate(PROBS)}})
        oof.to_csv(directory/'oof.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        scores = [r['metrics']['macro_f1'] for r in records]
        result = dict(scheme=scheme,variant=variant,cv_mean=float(np.mean(scores)),cv_std=float(np.std(scores,ddof=1)),
                      pooled=score(y,pred), oof_sha256=sha256_file(directory/'oof.csv.gz'),
                      all_converged=all(r['converged'] for r in records),
                      effective_parameters=pipe['model'].get_params(), fit_seconds=sum(r['fit_seconds'] for r in records))
        write(directory/'metrics.json', result)
        meta.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                    artifact_sha256={p.name:sha256_file(p) for p in directory.iterdir() if p.name!='metadata.json'})
        write(directory/'metadata.json',meta)
        return result
    except Exception as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write(directory/'metadata.json',meta)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','fit'])
    parser.add_argument('--data',default='data')
    parser.add_argument('--output',default='outputs/geo_group_cv_v1')
    parser.add_argument('--scheme',choices=['geo2','geo1'])
    parser.add_argument('--variant',choices=['C','G2'])
    parser.add_argument('--code-commit')
    args=parser.parse_args()
    if args.action=='prepare':
        print(json.dumps(prepare(args.data,args.output),ensure_ascii=False,indent=2))
    else:
        fit(args.data,args.output,args.scheme,args.variant,args.code_commit)

if __name__=='__main__':
    main()
