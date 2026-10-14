"""Independent audit of T13_geo2_te_confirm_v1; never fits a classifier.

- Manifests for seeds 42/101/202/303 are regenerated here with the T6 reference() procedure (integer tuple
  group lookup + StratifiedGroupKFold) and must match byte-for-byte (seed 42 = T5 36a40ddf...).
- LR_C / LGBM_O matrices must equal the unmodified C preprocessor refit preprocess-only on each training fold.
- LGBM_TE_G: validation TE equals the independent TE (T10 audit), training inner-OOF summary equals the
  independent group-inner OOF (T11 audit), non-geo columns equal the C reference.
- All 45 saved artifacts are SHA-checked, reloaded, and must reproduce the OOF probabilities.
- C1/C2 are recomputed with the T5 independent arithmetic and compared with the runner's analysis.
"""
from __future__ import annotations
import argparse
import hashlib
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
from experiments.geo_te_group_inner.audit import group_inner_oof
from experiments.geo2_te_confirm import run as r

ROOT = Path(__file__).resolve().parents[2]
HEAD, NTE = 28, 9
T5_GEO2_SHA = '36a40ddf1153b7a25bf36d5573b26b4dc9a0d67d240869c2e5cc1d85c1f89d64'


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def manifest_sha(X, y, ids, seed):
    cols = GEO[:2]
    tuples = [tuple(int(v) for v in row) for row in X[cols].itertuples(index=False, name=None)]
    lookup = {k: i for i, k in enumerate(sorted(set(tuples)))}
    g = np.array([lookup[t] for t in tuples])
    folds = np.full(len(X), -1, dtype=int)
    for f, (_, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g)):
        folds[b] = f
    frame = pd.DataFrame({'building_id': ids, 'fold': folds, **{c: X[c] for c in cols}})
    return hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest(), folds, g


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
    X, y, ids, folds, _ = r.verify(data, output)

    split_checks = []
    sha42, _, _ = manifest_sha(X, y, ids, 42)
    equal(sha42, T5_GEO2_SHA, 'seed 42 regenerated manifest == T5')
    for s in r.SEEDS:
        sha, f_ind, g = manifest_sha(X, y, ids, s)
        equal(sha, design['manifest_sha256'][str(s)], f'seed {s} manifest bytes')
        np.testing.assert_array_equal(f_ind, folds[s])
        for f in range(5):
            a, b = folds[s] != f, folds[s] == f
            need(not set(g[a]) & set(g[b]), 'group leakage')
            need(min(int((y[b] == k).sum()) for k in (1, 2, 3)) >= 30, 'class support')
            split_checks.append(dict(seed=s, fold=f, valid_rows=int(b.sum()), group_overlap=0))

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, a, f) for s in r.SEEDS for a in r.ARMS for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(len(ledger) == 45 and all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Ledger')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 45, 'fit_start events')
    need(all(e['started_at_utc'] > seal['created_at_utc'] for e in ledger.values()), 'Fit preceded design seal')
    for e in ledger.values():
        for n, h in e['artifact_sha256'].items():
            equal(digest(output/e['dir']/n), h, 'artifact '+n)
            equal(digest(output/'backup'/e['dir']/n), h, 'backup '+n)

    preds, fold_rows, models, checks = {}, [], [], []
    for s in r.SEEDS:
        fs = folds[s]
        oofs = {}
        for arm in r.ARMS:
            d = output/f's{s}'/arm
            equal(read_json(d/'metrics.json')['oof_sha256'], digest(d/'oof.csv.gz'), 'OOF SHA')
            oofs[arm] = check_oof(pd.read_csv(d/'oof.csv.gz', float_precision='round_trip'), ids, y, fs)
            preds[s, arm] = oofs[arm].y_pred.to_numpy()
        for f in range(5):
            a, b = np.flatnonzero(fs != f), np.flatnonzero(fs == f)
            with threadpool_limits(limits=2):
                ref = original('C')['preprocess'].fit(X.iloc[a])
                names_c = list(ref['columns'].get_feature_names_out())
                ref_valid = ref.transform(X.iloc[b]).tocsr()
            text_start = next(i for i, n in enumerate(names_c) if n.startswith('categorical__'+TEXT[0]))
            ytr = y.iloc[a].to_numpy()
            for arm in r.ARMS:
                d = output/f's{s}'/arm
                if arm == 'LR_C':
                    pipe = joblib.load(d/f'fold_{f}_model.joblib')  # SHA-verified, generated by this study
                    equal(pipe['model'].get_params(), original('C')['model'].get_params(), 'LR params')
                    with threadpool_limits(limits=1):
                        mv = pipe['preprocess'].transform(X.iloc[b]).tocsr()
                        need(mv.shape == ref_valid.shape and (mv != ref_valid).nnz == 0, 'LR_C matrix != C reference')
                        p = pipe.predict_proba(X.iloc[b])
                    need(pipe['model'].n_iter_.max() < pipe['model'].max_iter, 'LR nonconvergence')
                else:
                    pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
                    with threadpool_limits(limits=2):
                        mv = pre.transform(X.iloc[b]).tocsr()
                    if arm == 'LGBM_O':
                        need(mv.shape == ref_valid.shape and (mv != ref_valid).nnz == 0, 'LGBM_O matrix != C reference')
                    else:
                        step = pre['columns'].named_transformers_['geo_te']
                        equal(type(step).__module__, 'experiments.geo_te_group_inner.encoder', 'encoder module')
                        equal(step.group_level, 'geo2', 'inner group level')
                        need((mv[:, :HEAD] != ref_valid[:, :HEAD]).nnz == 0, 'TE_G head != C')
                        need((mv[:, HEAD+NTE:] != ref_valid[:, text_start:]).nnz == 0, 'TE_G text != C')
                        np.testing.assert_allclose(mv[:, HEAD:HEAD+NTE].toarray(),
                                                   te(X.iloc[a][GEO], ytr, X.iloc[b][GEO], m).astype(np.float32), atol=1e-7, rtol=0)
                        tr = group_inner_oof(X.iloc[a][GEO], ytr, 'geo2', m)
                        need((tr[:, 5] == 0).all(), 'training rows not unseen at geo2 level')
                        np.testing.assert_allclose(step.train_oof_mean_, tr.mean(0), atol=1e-12, rtol=0)
                        np.testing.assert_allclose(step.train_oof_std_, tr.std(0), atol=1e-12, rtol=0)
                    booster = lgb.Booster(model_file=str(d/f'fold_{f}_model.txt'))
                    params = model_params(d/f'fold_{f}_model.txt')
                    need(booster.current_iteration() == 300 and float(params['learning_rate']) == 0.05
                         and int(params['num_leaves']) == 31 and params.get('categorical_feature', '') == '', 'LGBM params')
                    p = booster.predict(mv)
                np.testing.assert_allclose(p, oofs[arm][PROBS].iloc[b].to_numpy(), atol=1e-12, rtol=0)
                calc = arithmetic(y.iloc[b], oofs[arm].y_pred.iloc[b])
                equal(read_json(d/f'fold_{f}.json')['metrics'], calc, 'fold metric vs independent arithmetic')
                checks.append(dict(seed=s, arm=arm, fold=f, max_prob_diff=float(np.abs(p-oofs[arm][PROBS].iloc[b].to_numpy()).max())))
                print('AUDIT', s, arm, f, 'PASS', flush=True)
        for arm in r.ARMS:
            vals = [arithmetic(y[fs == f], preds[s, arm][fs == f])['macro_f1'] for f in range(5)]
            np.testing.assert_allclose([read_json(output/f's{s}'/arm/'metrics.json')['cv_mean']], [np.mean(vals)], atol=1e-15, rtol=0)
            fold_rows += [dict(seed=s, arm=arm, fold=f, **arithmetic(y[fs == f], preds[s, arm][fs == f])) for f in range(5)]
            models.append(dict(seed=s, arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)), fold_macro_f1=vals,
                               prediction_share=[float(np.mean(preds[s, arm] == k)) for k in (1, 2, 3)], **arithmetic(y, preds[s, arm])))

    # Pooled over 15 folds (rows concatenated across seeds) and fold-level deltas.
    fv = {(x['seed'], x['arm']): x['fold_macro_f1'] for x in models}
    def deltas(a, b):
        return [p - q for s in r.SEEDS for p, q in zip(fv[s, a], fv[s, b])]
    yy = np.concatenate([y.to_numpy()]*3)
    pooled = {arm: arithmetic(yy, np.concatenate([preds[s, arm] for s in r.SEEDS])) for arm in r.ARMS}
    for arm in r.ARMS:
        vals = [v for s in r.SEEDS for v in fv[s, arm]]
        models.append(dict(seed='pooled15', arm=arm, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                           fold_macro_f1=vals, **pooled[arm]))
    rules = cfg['rules']; c1r, c2r = rules['C1'], rules['C2']
    d1, d2 = deltas('LGBM_TE_G', 'LGBM_O'), deltas('LGBM_TE_G', 'LR_C')
    repair = {}
    for s in r.SEEDS:
        acc = arithmetic(y, preds[s, 'LGBM_TE_G'])['accuracy'] - arithmetic(y, preds[s, 'LGBM_O'])['accuracy']
        agree = float(np.mean(preds[s, 'LGBM_TE_G'] == preds[s, 'LGBM_O']))
        repair[str(s)] = dict(accuracy_diff=acc, agreement=agree, holds=abs(acc) <= c1r['repair_every_seed']['abs_accuracy_diff_max']
                              and agree >= c1r['repair_every_seed']['prediction_agreement_min'])
    rec3 = pooled['LGBM_TE_G']['recall'][2] - pooled['LGBM_O']['recall'][2]
    c1 = (np.mean(d1) >= c1r['mean_delta_min'] and sum(x > 0 for x in d1) >= c1r['fold_wins_min']
          and rec3 >= c1r['pooled_grade3_recall_delta_min'] and all(v['holds'] for v in repair.values()))
    c2 = (np.mean(d2) > c2r['mean_delta_gt'] and sum(x > 0 for x in d2) >= c2r['fold_wins_min']
          and pooled['LGBM_TE_G']['recall'][2] >= pooled['LR_C']['recall'][2] + c2r['pooled_grade3_recall_margin'])
    rec = read_json(output/'analysis.json')['judgments']
    equal(rec['C1']['holds'], bool(c1), 'C1'); equal(rec['C2']['holds'], bool(c2), 'C2')
    equal(rec['confirmed_geo2_candidate'], bool(c1 and c2), 'confirmed')
    np.testing.assert_allclose([rec['C1']['inputs']['mean_delta'], rec['C2']['inputs']['mean_delta']], [np.mean(d1), np.mean(d2)], atol=1e-12, rtol=0)
    equal([rec['C1']['inputs']['fold_wins'], rec['C2']['inputs']['fold_wins']], [sum(x > 0 for x in d1), sum(x > 0 for x in d2)], 'wins')
    judgments = dict(C1=dict(holds=bool(c1), mean_delta=float(np.mean(d1)), fold_wins=int(sum(x > 0 for x in d1)),
                             fold_deltas=d1, pooled_grade3_recall_delta=rec3, repair=repair),
                     C2=dict(holds=bool(c2), mean_delta=float(np.mean(d2)), fold_wins=int(sum(x > 0 for x in d2)), fold_deltas=d2,
                             te_recall3=pooled['LGBM_TE_G']['recall'][2], lr_recall3=pooled['LR_C']['recall'][2]),
                     confirmed_geo2_candidate=bool(c1 and c2),
                     per_seed_delta_vs_O={str(s): float(np.mean(d1[i*5:(i+1)*5])) for i, s in enumerate(r.SEEDS)},
                     per_seed_delta_vs_LR_C={str(s): float(np.mean(d2[i*5:(i+1)*5])) for i, s in enumerate(r.SEEDS)})

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = [dict(seed=s, base=bb, new='LGBM_TE_G', group=n, **turnover(y[mk], preds[s, bb][mk], preds[s, 'LGBM_TE_G'][mk]))
              for s in r.SEEDS for bb in ['LGBM_O', 'LR_C'] for n, mk in masks.items()]

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
        seed42_regenerated_equals_T5=True, new_classifier_fits=45, fit_attempts=45, models_reloaded_full_validation=len(checks),
        max_reload_prob_diff=max(c['max_prob_diff'] for c in checks), group_leakage_checks=len(split_checks),
        backup_hash_checks=len(ledger), holdout_accessed=False, judgments=judgments,
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['Three new geo2 Group splits (ARI 0.33-0.39 vs seed 42) are not fully independent of each other or of seed 42.',
                     'Random-CV performance of LGBM_TE_G is the reused T10 stratified result.',
                     'Model reload is not an independent retraining; no holdout; no significance or causal claim.'])
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
    j = res['judgments']
    print(res['status'], 'C1', j['C1']['holds'], round(j['C1']['mean_delta'], 5), j['C1']['fold_wins'],
          'C2', j['C2']['holds'], round(j['C2']['mean_delta'], 5), j['C2']['fold_wins'], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
