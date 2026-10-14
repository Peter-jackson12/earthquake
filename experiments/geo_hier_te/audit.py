"""Independent audit of T10_geo_hier_te_v1; never fits a classifier.

The hierarchical target encoding is re-implemented here with pandas merges on integer tuple keys
(independent of the study's encoder). For every fold:
- validation TE features in the saved preprocessor must equal this implementation computed from the
  training fold only (validation labels unused);
- the saved encoder's training inner-OOF summary (mean/std) and, for LR, the fitted StandardScaler
  must equal those of this implementation's inner-OOF encodings (no self-label use);
- non-geo columns must equal the unmodified C preprocessor refit preprocess-only on the training fold;
- reloaded models must reproduce the OOF probabilities.
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from preprocess import GEO, TEXT
from experiments.age995.run import make_pipeline as original
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json, PROBS
from experiments.geo_ablation.audit import turnover
from experiments.geo_hier_te import run as r

ROOT = Path(__file__).resolve().parents[2]
HEAD = 28
NTE = 9
T8_PUBLIC = ROOT/'reports/lgbm_native_cat_20261014.json'


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def te(Gtr, ytr, Gq, m):
    """Independent hierarchical TE: (n*rate + m*parent)/(n+m); unseen -> parent, cnt 0."""
    ytr = np.asarray(ytr)
    tr = pd.DataFrame({c: Gtr[c].astype('int64').to_numpy() for c in GEO})
    tr['a'] = (ytr == 1).astype(float); tr['b'] = (ytr == 3).astype(float)
    q = pd.DataFrame({c: Gq[c].astype('int64').to_numpy() for c in GEO})
    pa, pb = tr['a'].mean(), tr['b'].mean()
    q['pa'], q['pb'] = pa, pb
    out = np.zeros((len(q), NTE))
    parent_tab = None
    for lvl in range(3):
        keys = list(GEO[:lvl+1])
        g = tr.groupby(keys).agg(n=('a', 'size'), sa=('a', 'sum'), sb=('b', 'sum')).reset_index()
        if parent_tab is None:
            g['ppa'], g['ppb'] = pa, pb
        else:
            g = g.merge(parent_tab.rename(columns={'ta': 'ppa', 'tb': 'ppb'})[keys[:-1]+['ppa', 'ppb']], on=keys[:-1], how='left')
            need(g[['ppa', 'ppb']].notna().all().all(), 'child without parent')
        g['ta'] = (g.sa + m*g.ppa)/(g.n + m); g['tb'] = (g.sb + m*g.ppb)/(g.n + m)
        j = q[keys].merge(g[keys+['n', 'ta', 'tb']], on=keys, how='left')
        seen = j['n'].notna().to_numpy()
        ta = np.where(seen, j['ta'].to_numpy(), q['pa'].to_numpy())
        tb = np.where(seen, j['tb'].to_numpy(), q['pb'].to_numpy())
        out[:, 3*lvl], out[:, 3*lvl+1] = ta, tb
        out[:, 3*lvl+2] = np.log1p(np.where(seen, j['n'].fillna(0).to_numpy(), 0.0))
        q['pa'], q['pb'] = ta, tb
        parent_tab = g[keys+['ta', 'tb']]
    return out


def inner_oof(Gtr, ytr, m):
    out = np.zeros((len(Gtr), NTE))
    for _, v in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(ytr)), ytr):
        t = np.setdiff1d(np.arange(len(ytr)), v)
        out[v] = te(Gtr.iloc[t], np.asarray(ytr)[t], Gtr.iloc[v], m)
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
    equal(m, 20.0, 'pre-registered smoothing')
    X, y, ids, folds, _ = r.verify(data, output)

    t8_pub = {(mm['scheme'], mm['arm']): mm['cv_mean'] for mm in read_json(T8_PUBLIC)['models']}
    base = {}
    for b, name in [('LR_C', 'LR_C'), ('LGBM_O', 'O')]:
        for s in r.SCHEMES:
            info = design['baselines'][b][s]
            equal(digest(info['path']), info['sha256'], f'baseline SHA {b}/{s}')
            base[s, b] = check_oof(pd.read_csv(info['path'], float_precision='round_trip'), ids, y, folds[s]).y_pred.to_numpy()
            fm = float(np.mean([arithmetic(y[folds[s] == f], base[s, b][folds[s] == f])['macro_f1'] for f in range(5)]))
            np.testing.assert_allclose([fm], [t8_pub[s, name]], atol=1e-12, rtol=0)

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, a, f) for a in r.ARMS for s in r.SCHEMES for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(len(ledger) == 30 and all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Ledger')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 30, 'fit_start events')
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

    lgbm_want = cfg['lgbm_params'] if 'lgbm_params' in cfg else None
    preds = dict(base); fold_rows, models, checks = [], [], []
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
                names_c = list(ref['columns'].get_feature_names_out())
                ref_valid = ref.transform(X.iloc[b]).tocsr()
            text_start = next(i for i, n in enumerate(names_c) if n.startswith('categorical__'+TEXT[0]))
            ytr = y.iloc[a].to_numpy()
            te_valid = te(X.iloc[a][GEO], ytr, X.iloc[b][GEO], m)
            te_train = inner_oof(X.iloc[a][GEO], ytr, m)
            for arm in r.ARMS:
                d = output/scheme/arm
                ent = ledger[r.key(scheme, arm, f)]['artifact_sha256']
                for n in ent:
                    equal(digest(d/n), ent[n], 'artifact SHA '+n)
                if arm == 'LR_TE':
                    pipe = joblib.load(d/f'fold_{f}_model.joblib')  # SHA-verified, generated by this study
                    pre = pipe['preprocess']
                else:
                    pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
                step = pre['columns'].named_transformers_['geo_te']
                enc = step['te'] if hasattr(step, 'named_steps') else step
                with threadpool_limits(limits=2):
                    mv = pre.transform(X.iloc[b]).tocsr()
                names = list(pre['columns'].get_feature_names_out())
                need(not any('geo_level' in n for n in names), 'raw geo column present')
                need((mv[:, :HEAD] != ref_valid[:, :HEAD]).nnz == 0, 'head != C reference')
                need((mv[:, HEAD+NTE:] != ref_valid[:, text_start:]).nnz == 0, 'text one-hot != C reference')
                block = mv[:, HEAD:HEAD+NTE].toarray()  # 9 columns only
                if arm == 'LR_TE':
                    sc = step['scale']
                    np.testing.assert_allclose(sc.mean_, te_train.astype(np.float32).astype(np.float64).mean(0), atol=1e-6, rtol=0)
                    np.testing.assert_allclose(sc.scale_, te_train.astype(np.float32).astype(np.float64).std(0), rtol=1e-5, atol=1e-7)
                    expect_valid = (te_valid.astype(np.float32) - sc.mean_)/sc.scale_
                    np.testing.assert_allclose(block, expect_valid, atol=1e-5, rtol=0)
                else:
                    np.testing.assert_allclose(block, te_valid.astype(np.float32), atol=1e-7, rtol=0)
                np.testing.assert_allclose(enc.train_oof_mean_, te_train.mean(0), atol=1e-12, rtol=0)
                np.testing.assert_allclose(enc.train_oof_std_, te_train.std(0), atol=1e-12, rtol=0)
                if arm == 'LR_TE':
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
                unseen = {lvl: int((te_valid[:, 3*i+2] == 0).sum()) for i, lvl in enumerate(['geo1', 'geo2', 'geo3'])}
                checks.append(dict(scheme=scheme, arm=arm, fold=f, valid_unseen=unseen,
                                   max_prob_diff=float(np.abs(p-oofs[arm][PROBS].iloc[b].to_numpy()).max())))
                print('AUDIT', scheme, arm, f, 'PASS', flush=True)
        for arm in ['LR_C', 'LR_TE', 'LGBM_O', 'LGBM_TE']:
            pred = preds[scheme, arm]
            vals = [arithmetic(y[fs == f], pred[fs == f])['macro_f1'] for f in range(5)]
            if arm in r.ARMS:
                np.testing.assert_allclose([read_json(output/scheme/arm/'metrics.json')['cv_mean']], [np.mean(vals)], atol=1e-15, rtol=0)
            fold_rows += [dict(scheme=scheme, arm=arm, fold=f, **arithmetic(y[fs == f], pred[fs == f])) for f in range(5)]
            models.append(dict(scheme=scheme, arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                               fold_macro_f1=vals, **arithmetic(y, pred)))

    by = {(mm['scheme'], mm['arm']): mm for mm in models}
    win = lambda s, a, b: int(sum(x > z for x, z in zip(by[s, a]['fold_macro_f1'], by[s, b]['fold_macro_f1'])))
    rec = read_json(output/'analysis.json')['judgments']
    judgments = {}
    for arm, fam in [('LR_TE', 'LR'), ('LGBM_TE', 'LGBM')]:
        bl = r.BASELINE_OF[arm]
        gd = by['geo2', arm]['cv_mean'] - by['geo2', bl]['cv_mean']
        sd = by['stratified', arm]['cv_mean'] - by['stratified', bl]['cv_mean']
        rd = by['geo2', arm]['recall'][2] - by['geo2', bl]['recall'][2]
        gen = gd >= 0.010 and win('geo2', arm, bl) >= 4 and sd >= -0.010 and rd >= -0.01
        ch = (by['stratified', arm]['cv_mean'] > 0.688687 and win('stratified', arm, 'LR_C') >= 4
              and by['geo2', arm]['cv_mean'] >= 0.565146 and win('geo2', arm, 'LR_C') >= 4
              and by['geo2', arm]['recall'][2] >= 0.5210823515922837)
        equal(rec[fam]['generalization_candidate'], bool(gen), fam+' generalization judgment')
        equal(rec[fam]['baseline_challenger_candidate'], bool(ch), fam+' challenger judgment')
        np.testing.assert_allclose([rec[fam]['generalization_inputs']['geo2_mean_delta'],
                                    rec[fam]['generalization_inputs']['stratified_mean_delta']], [gd, sd], atol=1e-12, rtol=0)
        judgments[fam] = dict(generalization_candidate=bool(gen), baseline_challenger_candidate=bool(ch),
                              geo2_delta_vs_own=gd, stratified_delta_vs_own=sd, geo2_recall3_delta_vs_own=rd,
                              wins={s: {'vs_own': win(s, arm, bl), 'vs_LR_C': win(s, arm, 'LR_C')} for s in r.SCHEMES})

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy(), 'non995': X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = [dict(scheme=s, base=bb, new=arm, group=n, **turnover(y[mk], preds[s, bb][mk], preds[s, arm][mk]))
              for s in r.SCHEMES for arm, bb in [('LR_TE', 'LR_C'), ('LGBM_TE', 'LGBM_O'), ('LGBM_TE', 'LR_C')]
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
        new_classifier_fits=30, fit_attempts=30, models_reloaded_full_validation=len(checks),
        independent_te_validation_checks=len(checks), independent_te_train_oof_checks=len(checks),
        max_reload_prob_diff=max(c['max_prob_diff'] for c in checks), group_leakage_checks=len(split_checks),
        backup_hash_checks=len(ledger), holdout_accessed=False, judgments=judgments,
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['Baselines reused from audited runs (not refit); two arms on the same folds; no significance or causal claim.',
                     'Training TE uses random inner folds, so in Group CV training rows see their own region while '
                     'validation rows do not (train/validation TE distribution shift; design property, not leakage).',
                     'Model reload is not an independent retraining; no holdout, no external validation.'])
    write_json(destination/'audit.json', result)
    keep = {'all', 'age995', 'non995', 'geo1=26', 'geo1=20', 'geo1=13'}
    write_json(public, dict(audit={k: v for k, v in result.items() if k != 'output_sha256'},
                            private_audit_sha256=result['output_sha256'], models=models, folds=fold_rows,
                            paired=[p for p in paired if p['group'] in keep or p['group'].startswith('fold=')]))
    return result


def main():
    # Study artifacts pickled the encoder as __main__.HierGeoTargetEncoder (runner ran via python -m);
    # register the same class there so the SHA-verified joblib files can be loaded.
    import sys
    sys.modules['__main__'].HierGeoTargetEncoder = r.HierGeoTargetEncoder
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--destination', required=True); p.add_argument('--public', required=True)
    a = p.parse_args()
    res = audit(a.data, a.output, a.destination, a.public)
    print(res['status'], json.dumps(res['judgments'])[:600], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
