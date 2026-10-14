"""Independent audit of T12_struct_features_v1; never fits a classifier.

Derived features are recomputed here from explicitly listed raw columns (not prefix matching) and
standardized with training-fold statistics. Every saved matrix must equal the unmodified C preprocessor
(refit preprocess-only on the training fold) followed by these derived columns. SHA-verified artifacts are
reloaded and must reproduce the OOF probabilities; rules are recomputed with the T5 independent arithmetic.
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
from preprocess import GEO
from experiments.age995.run import make_pipeline as original
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json, PROBS
from experiments.geo_ablation.audit import turnover
from experiments.struct_features import run as r

ROOT = Path(__file__).resolve().parents[2]
T8_PUBLIC = ROOT/'reports/lgbm_native_cat_20261014.json'
SUP = ['adobe_mud', 'mud_mortar_stone', 'stone_flag', 'cement_mortar_stone', 'mud_mortar_brick', 'cement_mortar_brick',
       'timber', 'bamboo', 'rc_non_engineered', 'rc_engineered', 'other']
WEAK = ['adobe_mud', 'mud_mortar_stone', 'stone_flag', 'mud_mortar_brick', 'timber', 'bamboo']
STRONG = ['cement_mortar_stone', 'cement_mortar_brick', 'rc_non_engineered', 'rc_engineered']
USE = ['agriculture', 'hotel', 'rental', 'institution', 'school', 'industry', 'health_post', 'gov_office', 'use_police', 'other']
NAMES = {'S': ['n_super', 'n_weak', 'n_strong', 'n_use'], 'D': ['height_per_floor', 'area_x_floors', 'families_per_floor']}
NAMES['SD'] = NAMES['S'] + NAMES['D']


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def derived(X, bundle):
    s = lambda cols, pre: X[[pre + c for c in cols]].to_numpy(dtype=float).sum(1)
    fl = X['count_floors_pre_eq'].to_numpy(dtype=float)
    cols = {'n_super': s(SUP, 'has_superstructure_'), 'n_weak': s(WEAK, 'has_superstructure_'),
            'n_strong': s(STRONG, 'has_superstructure_'), 'n_use': s(USE, 'has_secondary_use_'),
            'height_per_floor': X['height_percentage'].to_numpy(dtype=float)/np.maximum(fl, 1),
            'area_x_floors': X['area_percentage'].to_numpy(dtype=float)*fl,
            'families_per_floor': X['count_families'].to_numpy(dtype=float)/np.maximum(fl, 1)}
    return np.column_stack([cols[n] for n in NAMES[bundle]])


def model_params(path):
    return dict(re.findall(r'^\[([a-z_0-9]+): (.*)\]$', Path(path).read_text(encoding='utf-8'), flags=re.M))


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
    for b in ['S', 'D', 'SD']:
        equal(cfg['bundles'][b]['features'], NAMES[b], 'bundle '+b)
    X, y, ids, folds, _ = r.verify(data, output)

    t8 = {(x['scheme'], x['arm']): x['cv_mean'] for x in read_json(T8_PUBLIC)['models']}
    preds = {}
    for b, name in [('LR_C', 'LR_C'), ('LGBM_O', 'O')]:
        for s in r.SCHEMES:
            info = design['baselines'][b][s]
            equal(digest(info['path']), info['sha256'], f'baseline SHA {b}/{s}')
            preds[s, b] = check_oof(pd.read_csv(info['path'], float_precision='round_trip'), ids, y, folds[s]).y_pred.to_numpy()
            fm = float(np.mean([arithmetic(y[folds[s] == f], preds[s, b][folds[s] == f])['macro_f1'] for f in range(5)]))
            np.testing.assert_allclose([fm], [t8[s, name]], atol=1e-12, rtol=0)

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, a, f) for a in r.ARMS for s in r.SCHEMES for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(len(ledger) == 90 and all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Ledger')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 90, 'fit_start events')
    need(all(e['started_at_utc'] > seal['created_at_utc'] for e in ledger.values()), 'Fit preceded design seal')
    for e in ledger.values():
        r.check_saved(output, e)

    split_checks = []
    for scheme, cols in [('geo2', GEO[:2]), ('geo1', GEO[:1])]:
        g = pd.factorize(pd.MultiIndex.from_frame(X[cols].astype('int64')))[0]
        for f in range(5):
            a, b = folds[scheme] != f, folds[scheme] == f
            need(not set(g[a]) & set(g[b]), 'Group leakage')
            split_checks.append(dict(scheme=scheme, fold=f, valid_rows=int(b.sum()), group_overlap=0))

    fold_rows, models, checks = [], [], []
    for scheme in r.SCHEMES:
        fs = folds[scheme]
        oofs = {}
        for arm in r.ARMS:
            d = output/scheme/arm
            equal(read_json(d/'metrics.json')['oof_sha256'], digest(d/'oof.csv.gz'), 'OOF SHA')
            oofs[arm] = check_oof(pd.read_csv(d/'oof.csv.gz', float_precision='round_trip'), ids, y, fs)
            preds[scheme, arm] = oofs[arm].y_pred.to_numpy()
        for f in range(5):
            a, b = np.flatnonzero(fs != f), np.flatnonzero(fs == f)
            with threadpool_limits(limits=2):
                ref = original('C')['preprocess'].fit(X.iloc[a])
                ref_valid = ref.transform(X.iloc[b]).tocsr()
            nc = ref_valid.shape[1]
            for arm in r.ARMS:
                d = output/scheme/arm
                bundle = arm.split('_')[1]
                ent = ledger[r.key(scheme, arm, f)]['artifact_sha256']
                lr = (d/f'fold_{f}_model.joblib').exists()
                if lr:
                    pipe = joblib.load(d/f'fold_{f}_model.joblib')  # SHA-verified, generated by this study
                    pre = pipe['preprocess']
                    equal(pipe['model'].get_params(), original('C')['model'].get_params(), 'LR params')
                else:
                    pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
                mods = {type(st).__module__ for _, st, _ in pre['columns'].transformers_ if hasattr(st, '__module__')}
                need(not any(mm == '__main__' for mm in mods), '__main__ pickle')
                with threadpool_limits(limits=2):
                    mv = pre.transform(X.iloc[b]).tocsr()
                k = len(NAMES[bundle])
                need(mv.shape[1] == nc + k, 'width')
                need((mv[:, :nc] != ref_valid).nnz == 0, 'C columns changed')
                tr, va = derived(X.iloc[a], bundle), derived(X.iloc[b], bundle)
                mu, sd = tr.mean(0), tr.std(0)
                np.testing.assert_allclose(mv[:, nc:].toarray(), ((va-mu)/sd).astype(np.float32), atol=2e-5, rtol=1e-5)
                if lr:
                    p = pipe.predict_proba(X.iloc[b])
                    need(pipe['model'].n_iter_.max() < pipe['model'].max_iter, 'LR nonconvergence')
                else:
                    booster = lgb.Booster(model_file=str(d/f'fold_{f}_model.txt'))
                    params = model_params(d/f'fold_{f}_model.txt')
                    need(booster.current_iteration() == 300 and float(params['learning_rate']) == 0.05
                         and int(params['num_leaves']) == 31, 'LGBM params')
                    p = booster.predict(mv)
                np.testing.assert_allclose(p, oofs[arm][PROBS].iloc[b].to_numpy(), atol=1e-12, rtol=0)
                calc = arithmetic(y.iloc[b], oofs[arm].y_pred.iloc[b])
                equal(read_json(d/f'fold_{f}.json')['metrics'], calc, 'fold metric vs independent arithmetic')
                checks.append(dict(scheme=scheme, arm=arm, fold=f, artifacts=sorted(ent),
                                   max_prob_diff=float(np.abs(p-oofs[arm][PROBS].iloc[b].to_numpy()).max())))
                print('AUDIT', scheme, arm, f, 'PASS', flush=True)
        for arm in ['LR_C', 'LGBM_O'] + r.ARMS:
            pred = preds[scheme, arm]
            vals = [arithmetic(y[fs == f], pred[fs == f])['macro_f1'] for f in range(5)]
            if arm in r.ARMS:
                np.testing.assert_allclose([read_json(output/scheme/arm/'metrics.json')['cv_mean']], [np.mean(vals)], atol=1e-15, rtol=0)
            fold_rows += [dict(scheme=scheme, arm=arm, fold=f, **arithmetic(y[fs == f], pred[fs == f])) for f in range(5)]
            models.append(dict(scheme=scheme, arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                               fold_macro_f1=vals, **arithmetic(y, pred)))

    by = {(x['scheme'], x['arm']): x for x in models}
    win = lambda s, a, b: int(sum(p > q for p, q in zip(by[s, a]['fold_macro_f1'], by[s, b]['fold_macro_f1'])))
    rule = cfg['rules']['joint_improvement_candidate']; rec = read_json(output/'analysis.json')['judgments']; judgments = {}
    for arm in r.ARMS:
        bl = 'LR_C' if arm.startswith('LR') else 'LGBM_O'
        ds = {s: by[s, arm]['cv_mean'] - by[s, bl]['cv_mean'] for s in r.SCHEMES}
        c1 = ds['stratified'] >= rule['stratified_mean_delta_min'] and win('stratified', arm, bl) >= rule['stratified_fold_wins_min']
        c2 = ds['geo2'] >= rule['geo2_mean_delta_min'] and win('geo2', arm, bl) >= rule['geo2_fold_wins_min']
        c3 = all(by[s, arm]['recall'][2] - by[s, bl]['recall'][2] >= rule['pooled_grade3_recall_delta_min'] for s in ['stratified', 'geo2'])
        c4 = all(by[s, arm]['error32'] <= rule['pooled_error32_ratio_max']*by[s, bl]['error32'] for s in ['stratified', 'geo2'])
        cand = c1 and c2 and c3 and c4
        equal(rec[arm]['joint_improvement_candidate'], bool(cand), arm+' judgment')
        equal([rec[arm]['conditions'][k] for k in ['c1_stratified_mean_and_wins', 'c2_geo2_mean_and_wins', 'c3_grade3_recall', 'c4_error32']],
              [bool(c1), bool(c2), bool(c3), bool(c4)], arm+' conditions')
        judgments[arm] = dict(candidate=bool(cand), conditions=[bool(c1), bool(c2), bool(c3), bool(c4)], deltas=ds,
                              wins={s: win(s, arm, bl) for s in r.SCHEMES})

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = [dict(scheme=s, base=('LR_C' if arm.startswith('LR') else 'LGBM_O'), new=arm, group=n,
                   **turnover(y[mk], preds[s, 'LR_C' if arm.startswith('LR') else 'LGBM_O'][mk], preds[s, arm][mk]))
              for s in r.SCHEMES for arm in r.ARMS for n, mk in masks.items()]

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
        new_classifier_fits=90, fit_attempts=90, models_reloaded_full_validation=len(checks),
        independent_derived_feature_checks=len(checks), max_reload_prob_diff=max(c['max_prob_diff'] for c in checks),
        group_leakage_checks=len(split_checks), backup_hash_checks=len(ledger), holdout_accessed=False,
        candidates=[a for a, j in judgments.items() if j['candidate']], judgments=judgments,
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['6 comparisons on the same folds; baselines reused (not refit); no significance or causal claim.',
                     'Model reload is not an independent retraining; no holdout, no external validation.'])
    write_json(destination/'audit.json', result)
    write_json(public, dict(audit={k: v for k, v in result.items() if k != 'output_sha256'},
                            private_audit_sha256=result['output_sha256'], models=models, folds=fold_rows, paired=paired))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--destination', required=True); p.add_argument('--public', required=True)
    a = p.parse_args()
    res = audit(a.data, a.output, a.destination, a.public)
    print(res['status'], res['candidates'], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
