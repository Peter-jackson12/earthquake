"""T6S: supplementary Group-CV audit of T6R against the recovered T5 C evidence; never fits a classifier.

Recovered T5 private package (C geo2/geo1 OOF + 10 C models) is verified with the unchanged
T6 reference() (pinned SHA, manifest reconstruction, SHA-checked model reload, prediction match).
T6R run outputs are re-verified (seal, ledger, artifact hashes) and compared row by row with C
using the unchanged T6 turnover(). Writes private evidence to --destination, aggregates to --public.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.geo_group_cv.audit import arithmetic, check_oof, digest, equal, need, read_json, write_json
from experiments.geo_ablation.audit import reference, turnover
from experiments.geo_ablation_local_repro import run as r

ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = 'T6S_geo_ablation_c_group_supplement_v1'
SUMMARY = ROOT/'reports/geo_ablation_recovery_summary.json'
ZIP_SHA256 = '346948cc1c53a3e62bf7d40c61512ffd74813a59a577045f162a2269344a57e4'
REGIONS = ['26', '20', '13']


def tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def sums(package):
    """Every member listed in the package SHA256SUMS.txt must match."""
    package = Path(package)
    listed = {}
    for line in (package/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        h, name = line.split(None, 1)
        listed[name.lstrip('*').strip()] = h
    for name, h in listed.items():
        equal(digest(package/name), h, 'package member '+name)
    actual = {k.replace('\\', '/') for k in tree(package)} - {'SHA256SUMS.txt'}
    equal(sorted(actual), sorted(listed), 'package member set')
    return len(listed)


def audit(data, package, zip_path, t6r, destination, public):
    package, t6r, destination = Path(package), Path(t6r), Path(destination)
    need(not destination.exists() and not Path(public).exists(), 'Do not overwrite any prior audit')
    equal(digest(zip_path), ZIP_SHA256, 'source ZIP SHA')
    members = sums(package)
    t5, baseline = package/'outputs/geo_group_cv_v1', package/'baseline_private'
    before = {'package': tree(package), 't6r': tree(t6r)}

    ref = reference(data, t5, baseline, reload_models=True, reconstruct=True)
    X, y, folds = ref['X'], ref['y'], ref['assignments']

    Xr, yr, ids, folds_r, design = r.verify(data, t6r)
    np.testing.assert_array_equal(ids, ref['ids']); np.testing.assert_array_equal(yr, y)
    for s in r.SCHEMES:
        np.testing.assert_array_equal(folds_r[s], folds[s])
    ledger = read_json(t6r/'ledger.json')['folds']
    need(len(ledger) == 45 and all(e['status'] == 'complete' for e in ledger.values()), 'T6R ledger')
    for e in ledger.values():
        r.check_saved(t6r, e)

    preds = {k: v for k, v in ref['predictions'].items()}
    for s in r.SCHEMES:
        for v in r.VARIANTS:
            d = t6r/s/v
            equal(digest(d/'oof.csv.gz'), read_json(d/'metrics.json')['oof_sha256'], 'T6R OOF SHA')
            oof = check_oof(pd.read_csv(d/'oof.csv.gz', float_precision='round_trip'), ids, y, folds[s])
            preds[s, v] = oof.y_pred.to_numpy()

    # The T6R report used published C aggregates for Group CV; confirm they equal row-level C.
    published = read_json(SUMMARY)['baseline_comparisons']
    c_models = {m['scheme']: m for m in ref['models']}
    for s in ['geo2', 'geo1']:
        np.testing.assert_allclose([c_models[s]['cv_mean'], c_models[s]['cv_std']],
                                   [published[s]['cv_mean'], published[s]['cv_std_ddof1']], atol=1e-12, rtol=0)

    masks = {'all': np.ones(len(X), bool), 'age995': X.age.eq(995).to_numpy(), 'non995': X.age.ne(995).to_numpy()}
    masks.update({'geo1='+g: X.geo_level_1_id.astype(str).eq(g).to_numpy() for g in REGIONS})
    paired, wins, models = [], [], []
    for s in r.SCHEMES:
        fs = folds[s]
        groups = {**masks, **{f'fold={f}': fs == f for f in range(5)}}
        for v in r.VARIANTS:
            for n, m in groups.items():
                if m.any():
                    paired.append(dict(scheme=s, variant=v, group=n, **turnover(y[m], preds[s, 'C'][m], preds[s, v][m])))
            c_f = [arithmetic(y[fs == f], preds[s, 'C'][fs == f])['macro_f1'] for f in range(5)]
            v_f = [arithmetic(y[fs == f], preds[s, v][fs == f])['macro_f1'] for f in range(5)]
            d = [b - a for a, b in zip(c_f, v_f)]
            wins.append(dict(scheme=s, variant=v, fold_delta_macro_f1=d, folds_better=int(sum(x > 0 for x in d)),
                             mean_delta=float(np.mean(d)), min_delta=min(d), max_delta=max(d)))
        for v in ['C'] + r.VARIANTS:
            vals = [arithmetic(y[fs == f], preds[s, v][fs == f])['macro_f1'] for f in range(5)]
            models.append(dict(scheme=s, variant=v, cv_mean=float(np.mean(vals)), cv_std=float(np.std(vals, ddof=1)),
                               fold_macro_f1=vals, **arithmetic(y, preds[s, v])))

    after = {'package': tree(package), 't6r': tree(t6r)}
    equal(after, before, 'inputs unchanged by audit')
    contract = read_json(ROOT/'configs/data_contract.json')
    equal({n: digest(Path(data)/n) for n in contract['files']}, contract['files'], 'data preserved')

    destination.mkdir(parents=True)
    for name, rows in [('models', models), ('paired', paired), ('fold_wins', wins),
                       ('c_reference_folds', ref['fold_rows']), ('split_checks', ref['split_checks'])]:
        write_json(destination/(name+'.json'), rows)
    result = dict(study_id=STUDY_ID, status='PASS', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        relation='supplement to T6R (T6R_geo_ablation_local_repro_v1); C is the recovered T5 run, not refitted',
        audit_code_sha256=digest(__file__), t6r_design_sha256=digest(t6r/'design.json'),
        source_zip_sha256=ZIP_SHA256, package_members_verified=members,
        c_oof_sha256={m['scheme']: m['oof_sha256'] for m in ref['models']},
        c_models_reloaded=10, c_prediction_match_atol=1e-12, manifests_reconstructed=2,
        t6r_folds_hash_checked=len(ledger), classifier_fits=0, holdout_accessed=False,
        c_group_published_aggregate_match=True, data_sha256=contract['files'],
        output_sha256={p.name: digest(p) for p in destination.iterdir()},
        limitations=[
            'Recovered C models are reloaded (SHA-checked), not independently retrained.',
            'Development-only descriptive CV; no holdout, no significance test, no causal claim.'])
    write_json(destination/'audit.json', result)
    keep = {'all', 'age995', 'non995'} | {'geo1='+g for g in REGIONS}
    write_json(public, dict(audit={k: v for k, v in result.items() if k != 'output_sha256'},
                            private_audit_sha256=result['output_sha256'], models=models, fold_wins=wins,
                            paired=[p for p in paired if p['group'] in keep or p['group'].startswith('fold=')]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--package', required=True)
    p.add_argument('--zip', required=True); p.add_argument('--t6r', required=True)
    p.add_argument('--destination', required=True); p.add_argument('--public', required=True)
    a = p.parse_args()
    res = audit(a.data, a.package, a.zip, a.t6r, a.destination, a.public)
    print(res['status'], res['package_members_verified'], res['c_oof_sha256'])


if __name__ == '__main__':
    main()
