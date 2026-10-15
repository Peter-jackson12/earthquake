"""Independent audit of T8_lgbm_native_cat_v1; never fits a classifier.

Reference matrices are rebuilt from the unmodified C preprocessor refit preprocess-only on each
training fold. Arm O must equal it exactly; arm N must equal its 28 numeric/binary/flag columns plus
categorical codes recomputed here from training-fold vocabularies (unseen -> NaN). Saved LightGBM
model texts (SHA-verified) are reloaded as native Boosters and must reproduce the OOF probabilities.
Metrics use the T5 independent arithmetic(); turnover uses the unchanged T6 turnover().
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
from preprocess import CATEGORICAL, GEO
from experiments.age995.run import make_pipeline as original
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json, PROBS
from experiments.geo_ablation.audit import turnover
from experiments.lgbm_native_cat import run as r

ROOT = Path(__file__).resolve().parents[2]
HEAD = 28


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def model_params(path):
    text = Path(path).read_text(encoding='utf-8')
    params = dict(re.findall(r'^\[([a-z_0-9]+): (.*)\]$', text, flags=re.M))
    return params


def codes(train, valid):
    """Training-fold vocabulary codes, unseen -> NaN; computed independently of the study encoder."""
    out = np.full((len(valid), len(CATEGORICAL)), np.nan)
    for j, c in enumerate(CATEGORICAL):
        vocab = np.unique(train[c].to_numpy())
        v = valid[c].to_numpy()
        pos = np.searchsorted(vocab, v)
        pos_c = np.clip(pos, 0, len(vocab)-1)
        hit = vocab[pos_c] == v
        out[hit, j] = pos_c[hit]
    return out


def audit(data, output, destination, public):
    output, destination = Path(output), Path(destination)
    need(not destination.exists() and not Path(public).exists(), 'Do not overwrite any prior audit')
    before = tree(output)
    design = read_json(output/'design.json'); seal = read_json(output/'design_seal.json')
    equal(digest(output/'design.json'), seal['design_sha256'], 'design seal')
    need(design['classifiers_fitted'] == seal['classifiers_fitted'] == 0, 'Not a pre-fit design')
    need(not design['holdout_accessed'] and design['O_stratified_refits'] == 0, 'Design boundary')
    for n, h in design['source_sha256'].items():
        equal(digest(ROOT/n), h, 'source '+n)
    cfg = design['config']
    X, y, ids, folds, _ = r.verify(data, output)

    # Reference evidence anchors.
    t4 = cfg['t4_O_stratified_oof']
    equal(digest(design['t4_oof_path']), t4['sha256'], 'T4 OOF pinned SHA')
    ver = (ROOT/'reports/model_families_v1/verification.json').read_text(encoding='utf-8')
    need(t4['sha256'] in ver and t4['metrics_sha256'] in ver, 'T4 SHA anchor absent from verification.json')
    for s in r.SCHEMES:
        equal(digest(design['lr_c_oof_path'][s]), cfg['lr_c_oof_sha256'][s], 'LR-C OOF SHA '+s)

    ledger = read_json(output/'ledger.json')['folds']
    expected = {r.key(s, a, f) for a, ss in cfg['study_fits'].items() for s in ss for f in range(5)}
    equal(sorted(ledger), sorted(expected), 'ledger fold set')
    need(len(ledger) == cfg['max_new_study_fits'] == 25, 'Fit budget')
    need(all(e['status'] == 'complete' and e['attempts'] == 1 for e in ledger.values()), 'Incomplete/retried fold')
    events = [json.loads(l) for l in (output/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    equal(len([e for e in events if e.get('action') == 'fit_start']), 25, 'fit_start events')
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
    preds, fold_rows, models, checks = {}, [], [], []
    pre_fits = 0
    for scheme in r.SCHEMES:
        fs = folds[scheme]
        lr = check_oof(pd.read_csv(design['lr_c_oof_path'][scheme], float_precision='round_trip'), ids, y, fs)
        preds[scheme, 'LR_C'] = lr.y_pred.to_numpy()
        oofs = {}
        for arm in ['O', 'N']:
            if scheme in cfg['study_fits'][arm]:
                d = output/scheme/arm
                equal(read_json(d/'metrics.json')['oof_sha256'], digest(d/'oof.csv.gz'), 'OOF SHA')
                oofs[arm] = check_oof(pd.read_csv(d/'oof.csv.gz', float_precision='round_trip'), ids, y, fs)
        if scheme == 'stratified':
            oofs['O'] = check_oof(pd.read_csv(design['t4_oof_path'], float_precision='round_trip'), ids, y, fs)
        for f in range(5):
            a, b = np.flatnonzero(fs != f), np.flatnonzero(fs == f)
            with threadpool_limits(limits=2):
                ref = original('C')['preprocess'].fit(X.iloc[a]); pre_fits += 1
                ref_valid = ref.transform(X.iloc[b]).tocsr()
                st, sv = ref['schema'].transform(X.iloc[a]), ref['schema'].transform(X.iloc[b])
            for arm in ['O', 'N']:
                if scheme not in cfg['study_fits'][arm]:
                    continue
                d = output/scheme/arm
                ent = ledger[r.key(scheme, arm, f)]['artifact_sha256']
                for n in [f'fold_{f}_model.txt', f'fold_{f}_preprocessor.joblib']:
                    equal(digest(d/n), ent[n], 'artifact SHA '+n)
                pre = joblib.load(d/f'fold_{f}_preprocessor.joblib')  # SHA-verified, generated by this study
                with threadpool_limits(limits=2):
                    mv = pre.transform(X.iloc[b])
                if arm == 'O':
                    need((mv.tocsr() != ref_valid).nnz == 0 and mv.shape == ref_valid.shape, 'O matrix != C reference')
                else:
                    need(mv.shape == (len(b), HEAD+len(CATEGORICAL)), 'N width')
                    head = ref_valid[:, :HEAD].toarray()  # 28 dense columns only (not the full OHE matrix)
                    need(np.array_equal(mv[:, :HEAD], head.astype(np.float32)), 'N numeric/binary != C reference')
                    np.testing.assert_array_equal(mv[:, HEAD:], codes(st, sv).astype(np.float32))
                booster = lgb.Booster(model_file=str(d/f'fold_{f}_model.txt'))
                params = model_params(d/f'fold_{f}_model.txt')
                equal(booster.current_iteration(), want['n_estimators'], 'boosting rounds')
                for k, src in [('learning_rate', 'learning_rate'), ('num_leaves', 'num_leaves'), ('seed', 'random_state'),
                               ('min_data_in_leaf', 'min_child_samples'), ('max_bin', 'max_bin')]:
                    need(float(params[k]) == float(want[src]), 'LightGBM param '+k)
                cat_text = params.get('categorical_feature', '')
                if arm == 'N':
                    equal([int(i) for i in cat_text.split(',')], cfg['categorical_feature_indices_N'], 'categorical_feature')
                    for k, v in cfg['lightgbm_categorical_defaults_expected'].items():
                        need(float(params[k]) == float(v), 'categorical default '+k)
                else:
                    need(cat_text == '', 'O must have no categorical_feature')
                p = booster.predict(mv)
                oof = oofs[arm]
                np.testing.assert_allclose(p, oof[PROBS].iloc[b].to_numpy(), atol=1e-12, rtol=0)
                np.testing.assert_array_equal(p.argmax(1)+1, oof.y_pred.iloc[b].to_numpy())
                calc = arithmetic(y.iloc[b], oof.y_pred.iloc[b])
                equal(read_json(d/f'fold_{f}.json')['metrics'], calc, 'fold metric vs independent arithmetic')
                checks.append(dict(scheme=scheme, arm=arm, fold=f, max_prob_diff=float(np.abs(p-oof[PROBS].iloc[b].to_numpy()).max()),
                                   unseen_codes=int(np.isnan(mv[:, HEAD:]).sum()) if arm == 'N' else None))
                print('AUDIT', scheme, arm, f, 'PASS', flush=True)
        for arm in ['LR_C', 'O', 'N']:
            pred = preds[scheme, 'LR_C'] if arm == 'LR_C' else oofs[arm].y_pred.to_numpy()
            preds[scheme, arm] = pred
            vals = [arithmetic(y[fs == f], pred[fs == f])['macro_f1'] for f in range(5)]
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            if arm != 'LR_C' and not (scheme == 'stratified' and arm == 'O'):
                np.testing.assert_allclose([read_json(output/scheme/arm/'metrics.json')['cv_mean']], [mean], atol=1e-15, rtol=0)
            fold_rows += [dict(scheme=scheme, arm=arm, fold=f, **arithmetic(y[fs == f], pred[fs == f])) for f in range(5)]
            models.append(dict(scheme=scheme, arm=arm, cv_mean=mean, cv_std=sd, fold_macro_f1=vals, **arithmetic(y, pred)))
    np.testing.assert_allclose([m['cv_mean'] for m in models if m['scheme'] == 'stratified' and m['arm'] == 'O'],
                               [t4['cv_mean']], atol=1e-12, rtol=0)

    by = {(m['scheme'], m['arm']): m for m in models}
    win = lambda s, a, b: int(sum(x > z for x, z in zip(by[s, a]['fold_macro_f1'], by[s, b]['fold_macro_f1'])))
    rules = cfg['rules']; p1, p2 = rules['primary_representation'], rules['secondary_baseline_challenger']
    delta = by['stratified', 'N']['cv_mean'] - by['stratified', 'O']['cv_mean']
    primary = delta >= p1['mean_delta_N_minus_O_min'] and win('stratified', 'N', 'O') >= p1['fold_wins_min']
    secondary = (by['stratified', 'N']['cv_mean'] > p2['stratified_mean_N_gt']
                 and win('stratified', 'N', 'LR_C') >= p2['stratified_fold_wins_vs_LR_C_min']
                 and by['geo2', 'N']['cv_mean'] >= p2['geo2_mean_N_ge'])
    rec = read_json(output/'analysis.json')['judgments']
    equal(rec['primary_representation_improvement'], bool(primary), 'primary judgment')
    equal(rec['secondary_baseline_challenger_candidate'], bool(secondary), 'secondary judgment')
    np.testing.assert_allclose([rec['primary_inputs']['mean_delta']], [delta], atol=1e-12, rtol=0)
    judgments = dict(primary=bool(primary), secondary=bool(secondary), stratified_delta_N_O=delta,
                     wins={s: {'N>O': win(s, 'N', 'O'), 'N>LR_C': win(s, 'N', 'LR_C'), 'O>LR_C': win(s, 'O', 'LR_C')} for s in r.SCHEMES},
                     deltas={s: {'N-O': by[s, 'N']['cv_mean']-by[s, 'O']['cv_mean'], 'N-LR_C': by[s, 'N']['cv_mean']-by[s, 'LR_C']['cv_mean'],
                                 'O-LR_C': by[s, 'O']['cv_mean']-by[s, 'LR_C']['cv_mean']} for s in r.SCHEMES})

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy(), 'non995': X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in ['26', '20', '13']})
    paired = []
    for s in r.SCHEMES:
        groups = {**masks, **{f'fold={f}': folds[s] == f for f in range(5)}}
        for base, new in [('O', 'N'), ('LR_C', 'N'), ('LR_C', 'O')]:
            for n, m in groups.items():
                paired.append(dict(scheme=s, base=base, new=new, group=n, **turnover(y[m], preds[s, base][m], preds[s, new][m])))

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
        t4_oof_sha256=t4['sha256'], lr_c_oof_sha256=cfg['lr_c_oof_sha256'], new_classifier_fits=25, fit_attempts=25,
        models_reloaded_full_validation=len(checks), reference_c_preprocessor_only_fits=pre_fits,
        max_reload_prob_diff=max(c['max_prob_diff'] for c in checks), group_leakage_checks=len(split_checks),
        backup_hash_checks=len(ledger), holdout_accessed=False, judgments=judgments,
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=['Two arms on the same CV folds; no significance test, no causal claim, no external validation.',
                     'O stratified is the pinned T4 OOF (not refit); model reload is not an independent retraining.'])
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
    print(res['status'], json.dumps(res['judgments'])[:400], res['max_reload_prob_diff'])


if __name__ == '__main__':
    main()
