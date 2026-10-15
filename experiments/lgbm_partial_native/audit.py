"""Independent audit of T9_lgbm_partial_native_v1; never fits a classifier.

Reference matrix: the unmodified C preprocessor refit preprocess-only on each training fold.
Arm P must equal its 28 numeric/binary/flag columns, its geo one-hot columns (same order), and
text codes recomputed here from training-fold vocabularies (unseen -> NaN). Saved LightGBM model
texts (SHA-verified) are reloaded as native Boosters and must reproduce the OOF probabilities.
Reference OOFs (T4/T8/LR-C) are pinned to the audited T8 run and its public aggregate.
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from preprocess import GEO, TEXT
from experiments.age995.run import make_pipeline as original
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json, PROBS
from experiments.geo_ablation.audit import turnover
from experiments.lgbm_partial_native import run as r

ROOT = Path(__file__).resolve().parents[2]
HEAD = 28
T8_PUBLIC = ROOT/'reports/lgbm_native_cat_20261014.json'


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def model_params(path):
    return dict(re.findall(r'^\[([a-z_0-9]+): (.*)\]$', Path(path).read_text(encoding='utf-8'), flags=re.M))


def codes(train, valid, cols):
    out = np.full((len(valid), len(cols)), np.nan)
    for j, c in enumerate(cols):
        vocab = np.unique(train[c].to_numpy())
        v = valid[c].to_numpy()
        pos = np.clip(np.searchsorted(vocab, v), 0, len(vocab)-1)
        hit = vocab[pos] == v
        out[hit, j] = pos[hit]
    return out


def audit(data, output, destination, public):
    output, destination = Path(output), Path(destination)
    need(not destination.exists() and not Path(public).exists(), 'Do not overwrite any prior audit')
    before = tree(output)
    design = read_json(output/'design.json'); seal = read_json(output/'design_seal.json')
    equal(digest(output/'design.json'), seal['design_sha256'], 'design seal')
    need(design['classifiers_fitted'] == seal['classifiers_fitted'] == 0, 'Not a pre-fit design')
    need(not design['holdout_accessed'], 'Design boundary')
    for n, h in design['source_sha256'].items():
        equal(digest(ROOT/n), h, 'source '+n)
    cfg = design['config']
    X, y, ids, folds, _ = r.verify(data, output)

    # Reference anchors: T8 audited public aggregate means + T8 run metrics SHA + T4/LR-C pins.
    t8_pub = {(m['scheme'], m['arm']): m['cv_mean'] for m in read_json(T8_PUBLIC)['models']}
    t8_run = Path(design['reference_oof_path']['N']['stratified']).parents[2]
    refs = {}
    for arm in r.REFS:
        for s in r.SCHEMES:
            path = design['reference_oof_path'][arm][s]
            equal(digest(path), cfg['reference_oof_sha256'][arm][s], f'reference OOF SHA {arm}/{s}')
            if arm in ('O', 'N') and Path(path).is_relative_to(t8_run):
                equal(read_json(Path(path).parent/'metrics.json')['oof_sha256'], digest(path), 'T8 run anchor')
            refs[s, arm] = check_oof(pd.read_csv(path, float_precision='round_trip'), ids, y, folds[s]).y_pred.to_numpy()
            fm = float(np.mean([arithmetic(y[folds[s] == f], refs[s, arm][folds[s] == f])['macro_f1'] for f in range(5)]))
            np.testing.assert_allclose([fm], [t8_pub[s, arm]], atol=1e-12, rtol=0)

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, 'P', f) for s in r.SCHEMES for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Incomplete/retried fold')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 15, 'fit_start events')
    need(all(e['started_at_utc'] > seal['created_at_utc'] for e in ledger.values()), 'Fit preceded design seal')
    for e in ledger.values():
        r.check_saved(output, e)

    split_checks = []
    for scheme, cols in [('geo2', GEO[:2]), ('geo1', GEO[:1])]:
        g = pd.factorize(pd.MultiIndex.from_frame(X[cols].astype('int64')))[0]
        for f in range(5):
            a, b = folds[scheme] != f, folds[scheme] == f
            need(not set(g[a]) & set(g[b]), 'Group leakage across train/validation')
            split_checks.append(dict(scheme=scheme, fold=f, valid_rows=int(b.sum()), group_overlap=0))

    want = cfg['lgbm_params']
    preds, fold_rows, models, checks = dict(refs), [], [], []
    for scheme in r.SCHEMES:
        fs = folds[scheme]
        d = output/scheme/'P'
        equal(read_json(d/'metrics.json')['oof_sha256'], digest(d/'oof.csv.gz'), 'OOF SHA')
        oof = check_oof(pd.read_csv(d/'oof.csv.gz', float_precision='round_trip'), ids, y, fs)
        preds[scheme, 'P'] = oof.y_pred.to_numpy()
        for f in range(5):
            a, b = np.flatnonzero(fs != f), np.flatnonzero(fs == f)
            with threadpool_limits(limits=2):
                ref = original('C')['preprocess'].fit(X.iloc[a])
                names = list(ref['columns'].get_feature_names_out())
                ref_valid = ref.transform(X.iloc[b]).tocsr()
                st, sv = ref['schema'].transform(X.iloc[a]), ref['schema'].transform(X.iloc[b])
            geo_idx = [i for i, n in enumerate(names) if any(n.startswith('categorical__'+g+'_') for g in GEO)]
            ent = ledger[r.key(scheme, 'P', f)]['artifact_sha256']
            for n in [f'fold_{f}_model.txt', f'fold_{f}_preprocessor.joblib']:
                equal(digest(d/n), ent[n], 'artifact SHA '+n)
            pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
            with threadpool_limits(limits=2):
                mv = pre.transform(X.iloc[b]).tocsr()
            need(mv.shape[1] == HEAD + len(TEXT) + len(geo_idx), 'P width')
            need((mv[:, :HEAD] != ref_valid[:, :HEAD]).nnz == 0, 'P head != C reference')
            need((mv[:, HEAD+len(TEXT):] != ref_valid[:, geo_idx]).nnz == 0, 'P geo one-hot != C reference')
            code_dense = mv[:, HEAD:HEAD+len(TEXT)].toarray()  # 8 columns only
            np.testing.assert_array_equal(code_dense, codes(st, sv, list(TEXT)).astype(np.float32))
            booster = lgb.Booster(model_file=str(d/f'fold_{f}_model.txt'))
            params = model_params(d/f'fold_{f}_model.txt')
            equal(booster.current_iteration(), want['n_estimators'], 'boosting rounds')
            for k, src in [('learning_rate', 'learning_rate'), ('num_leaves', 'num_leaves'), ('seed', 'random_state'),
                           ('min_data_in_leaf', 'min_child_samples'), ('max_bin', 'max_bin')]:
                need(float(params[k]) == float(want[src]), 'LightGBM param '+k)
            equal([int(i) for i in params['categorical_feature'].split(',')], cfg['categorical_feature_indices_P'], 'categorical_feature')
            for k, v in cfg['lightgbm_categorical_defaults_expected'].items():
                need(float(params[k]) == float(v), 'categorical default '+k)
            p = booster.predict(mv)
            np.testing.assert_allclose(p, oof[PROBS].iloc[b].to_numpy(), atol=1e-12, rtol=0)
            np.testing.assert_array_equal(p.argmax(1)+1, oof.y_pred.iloc[b].to_numpy())
            calc = arithmetic(y.iloc[b], oof.y_pred.iloc[b])
            equal(read_json(d/f'fold_{f}.json')['metrics'], calc, 'fold metric vs independent arithmetic')
            checks.append(dict(scheme=scheme, fold=f, max_prob_diff=float(np.abs(p-oof[PROBS].iloc[b].to_numpy()).max()),
                               unseen_text_codes=int(np.isnan(code_dense).sum())))
            print('AUDIT', scheme, 'P', f, 'PASS', flush=True)
        for arm in ['LR_C', 'O', 'N', 'P']:
            pred = preds[scheme, arm]
            vals = [arithmetic(y[fs == f], pred[fs == f])['macro_f1'] for f in range(5)]
            if arm == 'P':
                np.testing.assert_allclose([read_json(d/'metrics.json')['cv_mean']], [np.mean(vals)], atol=1e-15, rtol=0)
            fold_rows += [dict(scheme=scheme, arm=arm, fold=f, **arithmetic(y[fs == f], pred[fs == f])) for f in range(5)]
            models.append(dict(scheme=scheme, arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                               fold_macro_f1=vals, **arithmetic(y, pred)))

    by = {(m['scheme'], m['arm']): m for m in models}
    win = lambda s, a, b: int(sum(x > z for x, z in zip(by[s, a]['fold_macro_f1'], by[s, b]['fold_macro_f1'])))
    rule = cfg['rules']['combined_candidate']
    share = {s: (by[s, 'P']['cv_mean']-by[s, 'O']['cv_mean'])/(by[s, 'N']['cv_mean']-by[s, 'O']['cv_mean']) for s in r.SCHEMES}
    cond = dict(stratified_mean_P_gt=by['stratified', 'P']['cv_mean'] > rule['stratified_mean_P_gt'],
                stratified_fold_wins_vs_LR_C=win('stratified', 'P', 'LR_C') >= rule['stratified_fold_wins_vs_LR_C_min'],
                geo2_mean_P_ge=by['geo2', 'P']['cv_mean'] >= rule['geo2_mean_P_ge'],
                geo2_fold_wins_vs_LR_C=win('geo2', 'P', 'LR_C') >= rule['geo2_fold_wins_vs_LR_C_min'],
                geo2_pooled_grade3_recall_P_ge=by['geo2', 'P']['recall'][2] >= rule['geo2_pooled_grade3_recall_P_ge'])
    rec = read_json(output/'analysis.json')['judgments']
    equal(rec['combined_candidate'], all(cond.values()), 'combined candidate judgment')
    equal(rec['combined_conditions'], cond, 'combined conditions')
    np.testing.assert_allclose([rec['decomposition'][s+'_share'] for s in r.SCHEMES], [share[s] for s in r.SCHEMES], atol=1e-12, rtol=0)
    judgments = dict(combined_candidate=all(cond.values()), conditions=cond, decomposition_share=share,
                     wins={s: {f'P>{b}': win(s, 'P', b) for b in ['O', 'N', 'LR_C']} for s in r.SCHEMES},
                     deltas={s: {f'P-{b}': by[s, 'P']['cv_mean']-by[s, b]['cv_mean'] for b in ['O', 'N', 'LR_C']} for s in r.SCHEMES})

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy(), 'non995': X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = [dict(scheme=s, base=b, new='P', group=n, **turnover(y[m], preds[s, b][m], preds[s, 'P'][m]))
              for s in r.SCHEMES for b in ['O', 'N', 'LR_C']
              for n, m in {**masks, **{f'fold={f}': folds[s] == f for f in range(5)}}.items()]

    equal(tree(output), before, 'run outputs unchanged by audit')
    contract = read_json(ROOT/'configs/data_contract.json')
    equal({n: digest(Path(data)/n) for n in contract['files']}, contract['files'], 'data preserved')

    destination.mkdir(parents=True)
    for name, rows in [('models', models), ('folds', fold_rows), ('paired', paired), ('judgments', judgments),
                       ('model_checks', checks), ('split_checks', split_checks)]:
        write_json(destination/(name+'.json'), rows)
    result = dict(study_id=r.STUDY_ID, status='PASS', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        code_commit=design['code_commit'], design_sha256=digest(output/'design.json'), audit_code_sha256=digest(__file__),
        environment=design['environment'], data_sha256=contract['files'], manifest_sha256=design['manifest_sha256'],
        reference_oof_sha256=cfg['reference_oof_sha256'], new_classifier_fits=15, fit_attempts=15,
        models_reloaded_full_validation=len(checks), reference_c_preprocessor_only_fits=len(checks),
        max_reload_prob_diff=max(c['max_prob_diff'] for c in checks), group_leakage_checks=len(split_checks),
        backup_hash_checks=len(ledger), holdout_accessed=False, judgments=judgments,
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['Reference arms O/N/LR-C reused from audited runs (not refit); same folds; no significance or causal claim.',
                     'Model reload is not an independent retraining; no holdout, no external validation.'])
    write_json(destination/'audit.json', result)
    keep = {'all', 'age995', 'non995', 'geo1=26', 'geo1=20', 'geo1=13'}
    write_json(public, dict(audit={k: v for k, v in result.items() if k != 'output_sha256'},
                            private_audit_sha256=result['output_sha256'], models=models, folds=fold_rows,
                            paired=[p for p in paired if p['group'] in keep or p['group'].startswith('fold=')]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--destination', required=True); p.add_argument('--public', required=True)
    a = p.parse_args()
    res = audit(a.data, a.output, a.destination, a.public)
    print(res['status'], json.dumps(res['judgments'])[:500], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
