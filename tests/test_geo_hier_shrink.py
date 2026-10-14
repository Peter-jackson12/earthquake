"""T7 synthetic contracts only: no real development data, no holdout access.

The equivalence tests here are synthetic sample checks, not study training runs.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.base import clone
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_modeling import synthetic
from preprocess import CATEGORICAL, GEO, TEXT
from experiments.age995.run import make_pipeline as original, PARAMS
from experiments.geo_hier_shrink import run as t7

ONES = dict(geo1=1.0, geo2=1.0, geo3=1.0, text=1.0)


def strip(names):
    return [n.split('__', 1)[1] for n in names]


class SplitOHETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, _, _, _ = synthetic(400)
        X.loc[[0, 17, 155], 'age'] = 995
        rng = np.random.default_rng(7)
        geo = X.geo_level_1_id.astype(int).to_numpy()
        y = np.clip(1 + (geo % 3) + (rng.random(400) < 0.2), 1, 3)
        import pandas as pd
        cls.y = pd.Series(y)
        cls.train, cls.valid, cls.ytr = X.iloc[:320].copy(), X.iloc[320:].copy(), cls.y.iloc[:320]
        cls.valid.loc[cls.valid.index[:5], 'geo_level_3_id'] = '99999'  # unseen fine region
        with threadpool_limits(limits=1):
            cls.C = original('C').fit(cls.train, cls.ytr)
            cls.S = t7.make_pipeline(ONES).fit(cls.train, cls.ytr)

    def test_all_ones_matrix_equals_C_same_order(self):
        for part in [self.train, self.valid]:
            a = self.C['preprocess'].transform(part)
            b = self.S['preprocess'].transform(part)
            self.assertTrue(sparse.isspmatrix_csr(b) and b.dtype == np.float32)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual((a != b).nnz, 0)
        self.assertEqual(strip(self.C['preprocess']['columns'].get_feature_names_out()),
                         strip(self.S['preprocess']['columns'].get_feature_names_out()))

    def test_all_ones_predictions_match_C_sample_check(self):
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(self.C.predict(self.valid), self.S.predict(self.valid))
            np.testing.assert_allclose(self.C.predict_proba(self.valid), self.S.predict_proba(self.valid),
                                       rtol=0, atol=1e-6)

    def test_block_weights_scale_only_target_blocks(self):
        for v in ['H3_025', 'H23_010', 'H23_050']:
            w = t7.config()['variants'][v]
            pre = t7.variant_pipeline(v)['preprocess'].fit(self.train)
            m = pre.transform(self.valid)
            names = pre['columns'].get_feature_names_out().tolist()
            t7.check_blocks(m, names, w)
            base = self.C['preprocess'].transform(self.valid)
            scale = np.ones(len(names), dtype=np.float32)
            for b, cols in t7.block_index(names).items():
                scale[cols] = np.float32(w[b])
            expected = base @ sparse.diags(scale)
            self.assertEqual(abs(expected - m).max(), 0)

    def test_check_blocks_rejects_wrong_weight(self):
        pre = t7.variant_pipeline('H3_050')['preprocess'].fit(self.train)
        m = pre.transform(self.valid)
        names = pre['columns'].get_feature_names_out().tolist()
        with self.assertRaises(ValueError):
            t7.check_blocks(m, names, dict(ONES))

    def test_unknown_fine_region_only_uses_coarse(self):
        pre = t7.variant_pipeline('H23_010')['preprocess'].fit(self.train)
        m = pre.transform(self.valid.iloc[:5])
        idx = t7.block_index(pre['columns'].get_feature_names_out().tolist())
        self.assertEqual(m[:, idx['geo3']].nnz, 0)
        self.assertEqual(m[:, idx['geo1']].nnz, 5)


class PolicyTests(unittest.TestCase):
    def test_six_preregistered_variants(self):
        c = t7.config()
        self.assertEqual(sorted(c['variants']), sorted(['H3_050', 'H3_025', 'H3_010', 'H23_050', 'H23_025', 'H23_010']))
        for v, w in c['variants'].items():
            self.assertEqual(w['geo1'], 1.0); self.assertEqual(w['text'], 1.0)
            if v.startswith('H3_'):
                self.assertEqual(w['geo2'], 1.0)
            else:
                self.assertEqual(w['geo2'], w['geo3'])
        self.assertEqual(c['max_new_classifier_fits'], 6*3*5)
        self.assertEqual(c['follow_up_rule']['geo2_fold_wins_min'], 4)
        self.assertFalse(c['holdout_access'])

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            t7.variant_pipeline('H1_050')

    def test_fresh_C_not_mutated_and_params_fixed(self):
        p = t7.variant_pipeline('H3_010')
        self.assertEqual(p['model'].get_params(), original('C')['model'].get_params())
        for k, v in PARAMS.items():
            self.assertEqual(p['model'].get_params()[k], v)
        self.assertEqual(original('C')['preprocess']['columns'].transformers[2][2], CATEGORICAL)
        self.assertIsNone(original('C')['preprocess']['columns'].transformer_weights)
        cols = p['preprocess']['columns']
        self.assertEqual([n for n, _, _ in cols.transformers], ['numeric', 'binary', 'geo1', 'geo2', 'geo3', 'text'])
        self.assertEqual([c for _, _, cs in cols.transformers[2:] for c in cs], GEO + TEXT)
        ref = original('C')['preprocess']['columns'].transformers[2][1].get_params()
        for _, enc, _ in cols.transformers[2:]:
            self.assertEqual(enc.get_params(), ref)

    def test_clone_keeps_weights(self):
        p = clone(t7.variant_pipeline('H23_025'))
        self.assertEqual(p['preprocess']['columns'].transformer_weights, t7.config()['variants']['H23_025'])

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            with t7.locked(d):
                with self.assertRaises(ValueError):
                    with t7.locked(d, timeout=0.2):
                        pass
            with t7.locked(d):
                pass

    def test_check_saved_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for root in [d/'s/v', d/'backup/s/v']:
                root.mkdir(parents=True)
                (root/'fold_0.json').write_text('{}')
            entry = dict(dir='s/v', artifact_sha256={'fold_0.json': t7.sha256_file(d/'s/v/fold_0.json')})
            t7.check_saved(d, entry)
            (d/'backup/s/v/fold_0.json').write_text('{"x":1}')
            with self.assertRaises(ValueError):
                t7.check_saved(d, entry)

    def test_turnover_from_T6_audit(self):
        from experiments.geo_ablation.audit import turnover
        r = turnover([3, 3, 1, 2], [2, 3, 1, 2], [3, 1, 1, 2])
        self.assertEqual(r['corrected_predictions'], 1)
        self.assertEqual(r['new_errors'], 1)


class SyntheticRunTests(unittest.TestCase):
    """Ledger/checkpoint/finalize path on synthetic rows (90 tiny synthetic fits, not study fits)."""
    def test_fit_resume_finalize_analysis(self):
        import pandas as pd
        X, _, ids, folds = synthetic(300)
        geo = X.geo_level_1_id.astype(int).to_numpy()
        y = pd.Series(1 + geo % 3)
        y.iloc[::7] = 2
        ids = pd.Series(np.arange(1, 301))
        assign = {'stratified': np.arange(300) % 5, 'geo2': (np.arange(300)//3) % 5, 'geo1': geo % 5}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'run'
            (out/'backup').mkdir(parents=True)
            t7.atomic_write(out/'ledger.json', dict(study_id=t7.STUDY_ID, folds={}))
            c_paths = {}
            for s in t7.SCHEMES:  # fake "C" OOF to compare against
                rng = np.random.default_rng(1)
                p = rng.dirichlet([1, 1, 1], 300)
                frame = pd.DataFrame({'building_id': ids, 'fold': assign[s], 'y_true': y,
                                      'y_pred': p.argmax(1)+1, 'p_grade_1': p[:, 0], 'p_grade_2': p[:, 1],
                                      'p_grade_3': p[:, 2]})
                c_paths[s] = str(Path(d)/f'c_{s}.csv.gz')
                frame.to_csv(c_paths[s], index=False, compression={'method': 'gzip', 'mtime': 0})
            design = dict(max_new_fits=90, code_commit='0'*40, c_oof_path=c_paths,
                          c_oof_sha256={s: t7.sha256_file(c_paths[s]) for s in t7.SCHEMES})
            t7.write(out/'design.json', design)
            orig_verify = t7.verify
            t7.verify = lambda data, output: (X, y, ids, assign, design)
            try:
                with threadpool_limits(limits=1):
                    for s in t7.SCHEMES:
                        for v in t7.VARIANTS:
                            for f in range(5):
                                t7.fit_fold(X, y, ids, assign, out, s, v, f, design)
                    before = (out/'geo2/H3_050/fold_0.joblib').stat().st_mtime_ns
                    e = t7.fit_fold(X, y, ids, assign, out, 'geo2', 'H3_050', 0, design)  # resume: no refit
                    self.assertEqual(e['attempts'], 1)
                    self.assertEqual((out/'geo2/H3_050/fold_0.joblib').stat().st_mtime_ns, before)
                    res = t7.finalize_all('unused', out)
                self.assertEqual(res['ledger']['complete'], 90)
                self.assertEqual(res['ledger']['retries'], 0)
                for s in t7.SCHEMES:
                    sch = res['schemes'][s]
                    self.assertEqual(len(sch['C']['fold_macro_f1']), 5)
                    for v in t7.VARIANTS:
                        r = sch['variants'][v]
                        oof = t7.read_pred(out/s/v/'oof.csv.gz')
                        np.testing.assert_array_equal(oof.fold, assign[s])
                        self.assertAlmostEqual(r['cv_mean'], np.mean(r['fold_macro_f1']))
                        self.assertEqual(r['fold_wins_vs_C'], sum(dd > 0 for dd in r['fold_delta_vs_C']))
                        self.assertEqual(r['turnover_vs_C']['n'], 300)
                        self.assertTrue((out/'backup'/s/v/'oof.csv.gz').exists())
                for v, r in res['follow_up_candidates'].items():
                    expect = (r['geo2_delta'] >= 0.010 and r['geo2_wins'] >= 4 and r['stratified_delta'] >= -0.010)
                    self.assertEqual(r['meets_rule'], expect)
                self.assertTrue((out/'backup/analysis.json').exists())
                # A crashed (running) fold cannot push attempts beyond the 90-fit budget.
                ledger = t7.read_ledger(out)
                ledger['folds']['geo1/H23_010/4']['status'] = 'running'
                t7.atomic_write(out/'ledger.json', ledger)
                with self.assertRaisesRegex(ValueError, 'budget'):
                    t7.fit_fold(X, y, ids, assign, out, 'geo1', 'H23_010', 4, design)
            finally:
                t7.verify = orig_verify


if __name__ == '__main__':
    unittest.main()
