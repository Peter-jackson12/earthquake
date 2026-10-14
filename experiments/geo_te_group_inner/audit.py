"""Independent audit of T11_geo_te_group_inner_v1; never fits a classifier.

Reuses the audited T10 independent TE implementation (experiments.geo_hier_te.audit.te, pandas merges on
integer tuple keys; not the study encoder). Group-aware inner folds are rebuilt here with sklearn's
StratifiedGroupKFold on composite '|'-joined keys built from integer geo columns. For every geo2/geo1 fold:
- validation TE features equal the independent TE from the training fold only (labels of validation unused);
- the encoder's training inner-OOF mean/std (and LR scaler) equal the independent group-inner OOF;
- every training row is unseen at the outer group level in its inner encoding (100%);
- non-geo columns equal the unmodified C preprocessor refit preprocess-only on the training fold;
- reloaded artifacts (importable encoder module, no __main__ pickles) reproduce the OOF probabilities.
Stratified results are the reused, audited T10 OOFs (SHA + public means).
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
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from preprocess import GEO, TEXT
from experiments.age995.run import make_pipeline as original
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json, PROBS
from experiments.geo_ablation.audit import turnover
from experiments.geo_hier_te.audit import te
from experiments.geo_te_group_inner import run as r

ROOT = Path(__file__).resolve().parents[2]
HEAD, NTE = 28, 9
T8_PUBLIC = ROOT/'reports/lgbm_native_cat_20261014.json'
T10_PUBLIC = ROOT/'reports/geo_hier_te_20261014.json'
LEVEL_IDX = {'geo2': 1, 'geo1': 0}


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def group_inner_oof(Gtr, ytr, level, m):
    g = Gtr[GEO[0]].astype('int64').astype(str)
    if level == 'geo2':
        g = g + '|' + Gtr[GEO[1]].astype('int64').astype(str)
    g = g.to_numpy(dtype=object)
    out = np.zeros((len(Gtr), NTE))
    seen = set()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for _, v in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(ytr)), ytr, g):
            t = np.setdiff1d(np.arange(len(ytr)), v)
            need(not set(g[t]) & set(g[v]), 'inner group overlap')
            out[v] = te(Gtr.iloc[t], np.asarray(ytr)[t], Gtr.iloc[v], m)
            seen.update(v.tolist())
    need(len(seen) == len(ytr), 'inner folds incomplete')
    return out


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
    m = float(cfg['te']['smoothing_m'])
    equal(m, 20.0, 'smoothing')
    X, y, ids, folds, _ = r.verify(data, output)

    # References: baselines (anchored to T8 public means), T10 arms (anchored to T10 public means).
    t8 = {(x['scheme'], x['arm']): x['cv_mean'] for x in read_json(T8_PUBLIC)['models']}
    t10 = {(x['scheme'], x['arm']): x['cv_mean'] for x in read_json(T10_PUBLIC)['models']}
    preds = {}
    for b, anchor in [('LR_C', lambda s: t8[s, 'LR_C']), ('LGBM_O', lambda s: t8[s, 'O'])]:
        for s in r.SCHEMES:
            info = design['baselines'][b][s]
            equal(digest(info['path']), info['sha256'], f'baseline SHA {b}/{s}')
            preds[s, b] = check_oof(pd.read_csv(info['path'], float_precision='round_trip'), ids, y, folds[s]).y_pred.to_numpy()
            fm = float(np.mean([arithmetic(y[folds[s] == f], preds[s, b][folds[s] == f])['macro_f1'] for f in range(5)]))
            np.testing.assert_allclose([fm], [anchor(s)], atol=1e-12, rtol=0)
    for a in ['LR_TE', 'LGBM_TE']:
        for s in r.SCHEMES:
            info = design['t10_arms'][a][s]
            equal(digest(info['path']), info['sha256'], f'T10 SHA {a}/{s}')
            preds[s, a] = check_oof(pd.read_csv(info['path'], float_precision='round_trip'), ids, y, folds[s]).y_pred.to_numpy()
            fm = float(np.mean([arithmetic(y[folds[s] == f], preds[s, a][folds[s] == f])['macro_f1'] for f in range(5)]))
            np.testing.assert_allclose([fm], [t10[s, a]], atol=1e-12, rtol=0)
    for arm, t10arm in [('LR_TE_G', 'LR_TE'), ('LGBM_TE_G', 'LGBM_TE')]:
        preds['stratified', arm] = preds['stratified', t10arm]  # pre-registered reuse (inner split identical by construction)

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, a, f) for a in r.ARMS for s in r.FIT_SCHEMES for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(len(ledger) == 20 and all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Ledger')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 20, 'fit_start events')
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

    fold_rows, models, checks = [], [], []
    for scheme in r.FIT_SCHEMES:
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
                names_c = list(ref['columns'].get_feature_names_out())
                ref_valid = ref.transform(X.iloc[b]).tocsr()
            text_start = next(i for i, n in enumerate(names_c) if n.startswith('categorical__'+TEXT[0]))
            ytr = y.iloc[a].to_numpy()
            te_valid = te(X.iloc[a][GEO], ytr, X.iloc[b][GEO], m)
            te_train = group_inner_oof(X.iloc[a][GEO], ytr, scheme, m)
            lvl = LEVEL_IDX[scheme]
            need((te_train[:, 3*lvl+2] == 0).all(), 'training rows not unseen at the outer group level')
            for arm in r.ARMS:
                d = output/scheme/arm
                ent = ledger[r.key(scheme, arm, f)]['artifact_sha256']
                for n in ent:
                    equal(digest(d/n), ent[n], 'artifact SHA '+n)
                if arm == 'LR_TE_G':
                    pipe = joblib.load(d/f'fold_{f}_model.joblib')  # SHA-verified, generated by this study
                    pre = pipe['preprocess']
                else:
                    pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
                step = pre['columns'].named_transformers_['geo_te']
                enc = step['te'] if hasattr(step, 'named_steps') else step
                equal(type(enc).__module__, 'experiments.geo_te_group_inner.encoder', 'encoder pickle module')
                equal(enc.group_level, scheme, 'inner group level')
                with threadpool_limits(limits=2):
                    mv = pre.transform(X.iloc[b]).tocsr()
                names = list(pre['columns'].get_feature_names_out())
                need(not any('geo_level' in n for n in names), 'raw geo column present')
                need((mv[:, :HEAD] != ref_valid[:, :HEAD]).nnz == 0, 'head != C reference')
                need((mv[:, HEAD+NTE:] != ref_valid[:, text_start:]).nnz == 0, 'text one-hot != C reference')
                block = mv[:, HEAD:HEAD+NTE].toarray()  # 9 columns only
                t32 = te_train.astype(np.float32).astype(np.float64)
                if arm == 'LR_TE_G':
                    sc = step['scale']
                    np.testing.assert_allclose(sc.mean_, t32.mean(0), atol=1e-6, rtol=0)
                    exp_scale = t32.std(0)
                    exp_scale[exp_scale == 0] = 1.0  # sklearn StandardScaler maps zero variance to scale 1
                    np.testing.assert_allclose(sc.scale_, exp_scale, rtol=1e-5, atol=1e-7)
                    np.testing.assert_allclose(block, (te_valid.astype(np.float32)-sc.mean_)/sc.scale_, atol=1e-5, rtol=0)
                else:
                    np.testing.assert_allclose(block, te_valid.astype(np.float32), atol=1e-7, rtol=0)
                np.testing.assert_allclose(enc.train_oof_mean_, te_train.mean(0), atol=1e-12, rtol=0)
                np.testing.assert_allclose(enc.train_oof_std_, te_train.std(0), atol=1e-12, rtol=0)
                if arm == 'LR_TE_G':
                    p = pipe.predict_proba(X.iloc[b])
                    need(pipe['model'].n_iter_.max() < pipe['model'].max_iter, 'LR nonconvergence')
                else:
                    booster = lgb.Booster(model_file=str(d/f'fold_{f}_model.txt'))
                    params = model_params(d/f'fold_{f}_model.txt')
                    need(booster.current_iteration() == 300 and float(params['learning_rate']) == 0.05
                         and int(params['num_leaves']) == 31 and params.get('categorical_feature', '') == '', 'LGBM params')
                    p = booster.predict(mv)
                np.testing.assert_allclose(p, oofs[arm][PROBS].iloc[b].to_numpy(), atol=1e-12, rtol=0)
                calc = arithmetic(y.iloc[b], oofs[arm].y_pred.iloc[b])
                equal(read_json(d/f'fold_{f}.json')['metrics'], calc, 'fold metric vs independent arithmetic')
                checks.append(dict(scheme=scheme, arm=arm, fold=f, train_unseen_outer_level=1.0,
                                   max_prob_diff=float(np.abs(p-oofs[arm][PROBS].iloc[b].to_numpy()).max())))
                print('AUDIT', scheme, arm, f, 'PASS', flush=True)

    for scheme in r.SCHEMES:
        fs = folds[scheme]
        for arm in ['LR_C', 'LR_TE', 'LR_TE_G', 'LGBM_O', 'LGBM_TE', 'LGBM_TE_G']:
            pred = preds[scheme, arm]
            vals = [arithmetic(y[fs == f], pred[fs == f])['macro_f1'] for f in range(5)]
            if arm in r.ARMS and scheme in r.FIT_SCHEMES:
                np.testing.assert_allclose([read_json(output/scheme/arm/'metrics.json')['cv_mean']], [np.mean(vals)], atol=1e-15, rtol=0)
            fold_rows += [dict(scheme=scheme, arm=arm, fold=f, **arithmetic(y[fs == f], pred[fs == f])) for f in range(5)]
            models.append(dict(scheme=scheme, arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                               fold_macro_f1=vals, prediction_share=[float(np.mean(pred == k)) for k in (1, 2, 3)],
                               reused_from_T10=(scheme == 'stratified' and arm in r.ARMS), **arithmetic(y, pred)))

    by = {(x['scheme'], x['arm']): x for x in models}
    win = lambda s, a, b: int(sum(p > q for p, q in zip(by[s, a]['fold_macro_f1'], by[s, b]['fold_macro_f1'])))
    rules = cfg['rules']; rec = read_json(output/'analysis.json')['judgments']; judgments = {}
    for arm, fam in [('LR_TE_G', 'LR'), ('LGBM_TE_G', 'LGBM')]:
        bl = r.BASELINE_OF[arm]
        repair = {}
        for s in ['geo2', 'geo1']:
            acc = by[s, arm]['accuracy'] - by[s, bl]['accuracy']
            agree = float(np.mean(preds[s, arm] == preds[s, bl]))
            repair[s] = dict(accuracy_diff=acc, agreement=agree,
                             holds=abs(acc) <= rules['distribution_repair']['abs_accuracy_diff_max']
                             and agree >= rules['distribution_repair']['pooled_prediction_agreement_min'])
        rep = all(v['holds'] for v in repair.values())
        gd = by['geo2', arm]['cv_mean'] - by['geo2', bl]['cv_mean']
        sd = by['stratified', arm]['cv_mean'] - by['stratified', bl]['cv_mean']
        rd = by['geo2', arm]['recall'][2] - by['geo2', bl]['recall'][2]
        gc = rules['generalization_candidate']
        gen = rep and gd >= gc['geo2_mean_delta_min'] and win('geo2', arm, bl) >= gc['geo2_fold_wins_min'] \
            and rd >= gc['geo2_pooled_grade3_recall_delta_min'] and sd >= gc['stratified_mean_delta_min']
        bc = rules['baseline_challenger']
        ch = (by['stratified', arm]['cv_mean'] > bc['stratified_mean_gt'] and win('stratified', arm, 'LR_C') >= bc['stratified_fold_wins_vs_LR_C_min']
              and by['geo2', arm]['cv_mean'] >= bc['geo2_mean_ge'] and win('geo2', arm, 'LR_C') >= bc['geo2_fold_wins_vs_LR_C_min']
              and by['geo2', arm]['recall'][2] >= bc['geo2_pooled_grade3_recall_ge'])
        equal(rec[fam]['distribution_repair'], rep, fam+' repair')
        equal(rec[fam]['generalization_candidate'], bool(gen), fam+' generalization')
        equal(rec[fam]['baseline_challenger_candidate'], bool(ch), fam+' challenger')
        np.testing.assert_allclose([rec[fam]['generalization_inputs']['geo2_mean_delta']], [gd], atol=1e-12, rtol=0)
        judgments[fam] = dict(distribution_repair=rep, repair=repair, generalization_candidate=bool(gen),
                              baseline_challenger_candidate=bool(ch), geo2_delta_vs_own=gd,
                              geo2_delta_margin_over_threshold=gd-gc['geo2_mean_delta_min'], geo2_recall3_delta_vs_own=rd,
                              stratified_delta_vs_own_reused_T10=sd,
                              wins={s: {'vs_own': win(s, arm, bl), 'vs_LR_C': win(s, arm, 'LR_C')} for s in r.SCHEMES},
                              fold_deltas_geo2=[p-q for p, q in zip(by['geo2', arm]['fold_macro_f1'], by['geo2', bl]['fold_macro_f1'])])

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy(), 'non995': X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = [dict(scheme=s, base=bb, new=arm, group=n, **turnover(y[mk], preds[s, bb][mk], preds[s, arm][mk]))
              for s in r.FIT_SCHEMES for arm, bb in [('LR_TE_G', 'LR_C'), ('LR_TE_G', 'LR_TE'), ('LGBM_TE_G', 'LGBM_O'),
                                                     ('LGBM_TE_G', 'LGBM_TE'), ('LGBM_TE_G', 'LR_C')]
              for n, mk in {**masks, **{f'fold={f}': folds[s] == f for f in range(5)}}.items()]

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
        new_classifier_fits=20, fit_attempts=20, models_reloaded_full_validation=len(checks),
        independent_te_validation_checks=len(checks), independent_group_inner_oof_checks=len(checks),
        max_reload_prob_diff=max(c['max_prob_diff'] for c in checks), group_leakage_checks=len(split_checks),
        backup_hash_checks=len(ledger), holdout_accessed=False, stratified='reused audited T10 OOF (pre-registered)',
        judgments=judgments, output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['LGBM geo2 margin over the +0.010 threshold is ~1.3e-4 with 4/5 fold wins: borderline; same folds as T10 '
                     'and baselines -> optimistic if selecting; needs a separate pre-registered confirmation.',
                     'Stratified values are reused T10 results, not refit.',
                     'Model reload is not an independent retraining; no holdout, no significance or causal claim.'])
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
    print(res['status'], json.dumps(res['judgments'])[:900], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
