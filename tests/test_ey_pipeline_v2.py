"""T35 synthetic contracts for `experiments/ey_pipeline_v2/features.py` and `original_check.py`.

Synthetic frames only: no real development data, no holdout access, no model fitting.  Tests that need the private
notebook copy (kept outside Git) are skipped when it is absent, so unittest reports pass and skip separately.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import ALLOWED_TEXT, BINARY, FEATURES, GEO, NUMERIC, sha256_file  # noqa: E402
from experiments.ey_pipeline_v2 import features as F  # noqa: E402
from experiments.ey_pipeline_v2 import original_check as OC  # noqa: E402

PRIVATE = ROOT.parents[1]/'local_reproduction'/'T35_20261016_ey_pipeline_v2'/'source_copy'
NOTEBOOK = PRIVATE/'지진.ipynb'
NOTEBOOK_SHA256 = '633f6e6b3c1c3bdf669af71df29d31581cb97df63a303ab72b0bbc21b8eed33d'
HAVE_PRIVATE = NOTEBOOK.is_file()


def blank(n):
    """n identical rows in the notebook dtype layout (38 raw columns in FEATURES order, no label)."""
    d = {}
    for c in FEATURES:
        if c in GEO:
            d[c] = np.ones(n, dtype=np.int64)
        elif c in NUMERIC or c in BINARY:
            d[c] = np.zeros(n, dtype=np.int64)
        else:
            d[c] = [ALLOWED_TEXT[c][0]] * n
    for c in ('count_floors_pre_eq', 'age'):
        d[c] = np.ones(n, dtype=np.int64)
    return pd.DataFrame(d)[FEATURES]


def synth(n, seed=3, regions=5, with_label=True):
    """Random frame; geo level 1 has few values so every level-1 region is large, levels 2/3 mix thick and thin."""
    rng = np.random.default_rng(seed)
    g1 = rng.integers(0, regions, n)
    g2 = g1 * 100 + rng.integers(0, 7, n)
    g3 = g2 * 50 + rng.integers(0, 9, n)
    d = {'geo_level_1_id': g1, 'geo_level_2_id': g2, 'geo_level_3_id': g3,
         'count_floors_pre_eq': rng.integers(1, 6, n), 'age': rng.choice([0, 5, 10, 20, 30, 995], n),
         'area_percentage': rng.integers(1, 30, n), 'height_percentage': rng.integers(1, 12, n),
         'count_families': rng.integers(0, 4, n)}
    for c, vals in ALLOWED_TEXT.items():
        d[c] = rng.choice(vals, n)
    for c in BINARY:
        d[c] = (rng.random(n) < 0.15).astype(np.int64)
    fr = pd.DataFrame(d)[FEATURES]
    for c in GEO + NUMERIC:
        fr[c] = fr[c].astype(np.int64)
    if with_label:
        score = fr['age'].clip(upper=60) / 30 + fr['has_superstructure_adobe_mud'] + rng.normal(0, .8, n) + (g1 % 3) * .4
        fr[F.GRADE] = (np.digitize(score, np.quantile(score, [.25, .7])) + 1).astype(np.int64)
    return fr


class ColumnContractTests(unittest.TestCase):
    def test_75_columns_with_documented_order(self):
        cols = F.feature_columns()
        self.assertEqual(len(cols), 75)
        self.assertEqual(len(set(cols)), 75)
        self.assertEqual(cols[:38], list(FEATURES))
        derived = ['material_group', 'structure_type', 'n_weak_material', 'n_material', 'weak_age']
        for lv in (1, 2, 3):
            derived += [f'g{lv}_n_buildings', f'g{lv}_weak_ratio', f'g{lv}_rc_ratio', f'g{lv}_age_vs_region']
        for name in ['region_material', 'region_structure', 'material_structure', 'region_material_floor',
                     'g2', 'g3', 'g2_material', 'g3_material']:
            derived += [f'{name}_high_rate', f'{name}_low_rate']
        derived += ['g2_material_high_gap', 'g2_material_low_gap', 'g3_material_high_gap', 'g3_material_low_gap']
        self.assertEqual(len(derived), 37)
        self.assertEqual(F.derived_columns(), derived)
        self.assertEqual(cols[38:], derived)
        self.assertNotIn('material_high_rate', cols)    # k_material feeds only the gap columns


class BuildingFeatureTests(unittest.TestCase):
    def test_material_precedence_and_sums_by_hand(self):
        d = blank(8)
        d.loc[0, 'has_superstructure_other'] = 1                                                    # other
        d.loc[1, ['has_superstructure_timber', 'has_superstructure_bamboo']] = 1                    # timber_bamboo
        d.loc[2, ['has_superstructure_timber', 'has_superstructure_adobe_mud']] = 1                 # mud_stone > timber
        d.loc[3, ['has_superstructure_stone_flag', 'has_superstructure_mud_mortar_brick']] = 1      # mud_stone, 2 weak
        d.loc[4, ['has_superstructure_adobe_mud', 'has_superstructure_cement_mortar_stone']] = 1    # cement > mud
        d.loc[5, ['has_superstructure_timber', 'has_superstructure_cement_mortar_brick']] = 1       # cement > timber
        d.loc[6, ['has_superstructure_cement_mortar_brick', 'has_superstructure_rc_non_engineered']] = 1  # RC > cement
        d.loc[7, ['has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
                  'has_superstructure_rc_engineered', 'has_superstructure_other']] = 1              # RC wins; n=4
        out = F.building_features(d)
        self.assertEqual(out['material_group'].tolist(),
                         ['other', 'timber_bamboo', 'mud_stone', 'mud_stone', 'cement', 'cement', 'RC', 'RC'])
        self.assertEqual(out['n_weak_material'].tolist(), [0, 0, 1, 2, 1, 0, 0, 2])
        self.assertEqual(out['n_material'].tolist(), [1, 2, 2, 2, 2, 2, 2, 4])
        self.assertEqual(out.loc[:, 'has_superstructure_adobe_mud'].tolist(), d['has_superstructure_adobe_mud'].tolist())
        self.assertNotIn('material_group', d.columns)                                              # input not mutated

    def test_no_superstructure_flag_is_other(self):
        out = F.building_features(blank(2))
        self.assertEqual(out['material_group'].tolist(), ['other', 'other'])
        self.assertEqual(out['n_material'].tolist(), [0, 0])

    def test_floor_band_weak_age_and_keys_by_hand(self):
        d = blank(5)
        d['count_floors_pre_eq'] = [1, 2, 3, 4, 9]
        d['age'] = [10, 20, 30, 40, 995]
        d['geo_level_1_id'] = [3, 3, 4, 4, 5]
        d['geo_level_2_id'] = [301, 302, 401, 402, 501]
        d['geo_level_3_id'] = [30101, 30201, 40101, 40201, 50101]
        d.loc[[0, 4], 'has_superstructure_adobe_mud'] = 1             # mud_stone rows (age kept)
        d.loc[1, 'has_superstructure_rc_engineered'] = 1               # RC row (age zeroed)
        d['foundation_type'] = ['h', 'r', 'u', 'w', 'i']
        d['roof_type'] = ['n', 'q', 'x', 'n', 'q']
        d['ground_floor_type'] = ['f', 'm', 'v', 'x', 'z']
        out = F.building_features(d)
        self.assertEqual(out['floor_band'].tolist(), ['1', '2', '3', '3', '3'])
        self.assertTrue(all(isinstance(v, str) for v in out['floor_band']))
        self.assertEqual(out['weak_age'].tolist(), [10, 0, 0, 0, 995])
        self.assertEqual(out['structure_type'].tolist(), ['h_n_f', 'r_q_m', 'u_x_v', 'w_n_x', 'i_q_z'])
        self.assertEqual(out['k_region_material'].tolist(),
                         ['3|mud_stone', '3|RC', '4|other', '4|other', '5|mud_stone'])
        self.assertEqual(out['k_region_structure'].tolist(),
                         ['3|h_n_f', '3|r_q_m', '4|u_x_v', '4|w_n_x', '5|i_q_z'])
        self.assertEqual(out['k_material_structure'].tolist(),
                         ['mud_stone|h_n_f', 'RC|r_q_m', 'other|u_x_v', 'other|w_n_x', 'mud_stone|i_q_z'])
        self.assertEqual(out['k_region_material_floor'].tolist(),
                         ['3|mud_stone|1', '3|RC|2', '4|other|3', '4|other|3', '5|mud_stone|3'])
        self.assertEqual(out['k_g2'].tolist(), ['301', '302', '401', '402', '501'])
        self.assertEqual(out['k_g3'].tolist(), ['30101', '30201', '40101', '40201', '50101'])
        self.assertEqual(out['k_g2_material'].tolist(), ['301|mud_stone', '302|RC', '401|other', '402|other', '501|mud_stone'])
        self.assertEqual(out['k_g3_material'].tolist(),
                         ['30101|mud_stone', '30201|RC', '40101|other', '40201|other', '50101|mud_stone'])
        self.assertEqual(out['k_material'].tolist(), ['mud_stone', 'RC', 'other', 'other', 'mud_stone'])


class RegionCompositionTests(unittest.TestCase):
    @staticmethod
    def base_region_frame():
        rows = [dict(g=7, material_group='mud_stone' if i < 12 else ('RC' if i < 15 else 'other'), age=i) for i in range(30)]
        rows += [dict(g=9, material_group='mud_stone', age=100 + i) for i in range(5)]
        d = pd.DataFrame(rows)
        return d.rename(columns={'g': 'geo_level_1_id'})

    def test_thin_region_gets_base_wide_values_by_hand(self):
        base = self.base_region_frame()
        s = F.region_summary(base, 'geo_level_1_id')
        self.assertEqual(s['n'].to_dict(), {7: 30, 9: 5})
        self.assertAlmostEqual(s.loc[7, 'weak'], 12 / 30)
        self.assertAlmostEqual(s.loc[7, 'rc'], 3 / 30)
        self.assertEqual(s.loc[7, 'med_age'], 14.5)
        base_weak, base_rc, base_median = 17 / 35, 3 / 35, 17.0     # 35 rows: 12 + 5 mud_stone, 3 RC; sorted age[17] = 17
        self.assertEqual(float(base['age'].median()), base_median)
        self.assertAlmostEqual(s.loc[9, 'weak'], base_weak)          # 5 rows (< 20): base-wide, not its own 100%
        self.assertAlmostEqual(s.loc[9, 'rc'], base_rc)
        self.assertEqual(s.loc[9, 'med_age'], base_median)
        self.assertEqual(int(s.loc[9, 'n']), 5)                      # count itself stays the real count

    def test_exactly_min_region_rows_keep_their_own_values(self):
        base = pd.DataFrame({'geo_level_1_id': [1] * 20 + [2] * 20, 'age': [10] * 20 + [50] * 20,
                             'material_group': ['RC'] * 20 + ['other'] * 20})
        s = F.region_summary(base, 'geo_level_1_id')
        self.assertEqual(s.loc[1, 'rc'], 1.0)
        self.assertEqual(s.loc[1, 'med_age'], 10.0)
        self.assertEqual(s.loc[2, 'rc'], 0.0)
        self.assertEqual(s.loc[2, 'med_age'], 50.0)

    def test_region_absent_from_base_by_hand(self):
        base = self.base_region_frame()
        s = F.region_summary(base, 'geo_level_1_id')
        ev = pd.DataFrame({'geo_level_1_id': [7, 11], 'age': [20, 30]})
        F.attach_region(ev, 'geo_level_1_id', 1, s, 17.0)
        self.assertEqual(ev['g1_n_buildings'].tolist(), [30, 0])
        self.assertAlmostEqual(ev.loc[0, 'g1_weak_ratio'], 12 / 30)
        # unseen region: plain mean over the region-table rows (post-replacement), not the base-wide share
        self.assertAlmostEqual(ev.loc[1, 'g1_weak_ratio'], (12 / 30 + 17 / 35) / 2)
        self.assertAlmostEqual(ev.loc[1, 'g1_rc_ratio'], (3 / 30 + 3 / 35) / 2)
        self.assertAlmostEqual(ev.loc[0, 'g1_age_vs_region'], 20 - 14.5)
        self.assertAlmostEqual(ev.loc[1, 'g1_age_vs_region'], 30 - 17.0)    # age minus the base median

    def test_unseen_region_through_make_features(self):
        base = synth(400, seed=11, regions=4)
        ev = synth(40, seed=12, regions=4, with_label=False)
        ev.loc[0, ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']] = [77, 7700, 770000]
        ev.loc[0, 'age'] = 50
        feats_b, (feats_e,) = F.make_features(base, [ev])
        bf = F.building_features(base)
        base_age_median = float(bf['age'].median())
        row = feats_e.iloc[0]
        for lv, geo in ((1, 'geo_level_1_id'), (2, 'geo_level_2_id'), (3, 'geo_level_3_id')):
            work = bf.groupby(geo).agg(n=('age', 'size'), med=('age', 'median'))
            weak = bf.assign(w=bf.material_group == 'mud_stone').groupby(geo)['w'].mean()
            thin = work['n'] < F.MIN_REGION
            table_weak = weak.where(~thin, (bf.material_group == 'mud_stone').mean())
            self.assertEqual(row[f'g{lv}_n_buildings'], 0)
            self.assertAlmostEqual(row[f'g{lv}_weak_ratio'], float(table_weak.mean()), places=12)
            self.assertAlmostEqual(row[f'g{lv}_age_vs_region'], 50 - base_age_median, places=12)
        self.assertTrue((feats_b['g1_n_buildings'] > 0).all())


class RateTableTests(unittest.TestCase):
    def test_rate_table_by_hand_with_thin_keys(self):
        grade = [3] * 30 + [1] * 12 + [2] * 18 + [3] * 10          # key 'a': 60 rows, key 'b': 10 rows (all grade 3)
        data = pd.DataFrame({'k': ['a'] * 60 + ['b'] * 10, 'damage_grade': grade})
        t = F.rate_table(data, 'k', 50)
        self.assertEqual(t['n'].to_dict(), {'a': 60, 'b': 10})
        self.assertAlmostEqual(t.loc['a', 'high'], 0.5)
        self.assertAlmostEqual(t.loc['a', 'low'], 0.2)
        self.assertAlmostEqual(t.loc['b', 'high'], 40 / 70)          # thin -> data-wide, not its own 100%
        self.assertAlmostEqual(t.loc['b', 'low'], 12 / 70)
        t10 = F.rate_table(data, 'k', 10)                             # exactly min_count: own rate kept
        self.assertAlmostEqual(t10.loc['b', 'high'], 1.0)
        self.assertAlmostEqual(t10.loc['b', 'low'], 0.0)

    def test_eval_rates_come_from_the_full_base_table(self):
        base = synth(600, seed=21, regions=4)
        # g3 = 8000001: 3 rows (thin, min 10); g3 = 8000002: 12 rows, all grade 3; ids outside the random range
        extra = blank(15)
        extra['geo_level_1_id'] = 1
        extra['geo_level_2_id'] = 100
        extra['geo_level_3_id'] = [8000001] * 3 + [8000002] * 12
        extra[F.GRADE] = [1, 1, 1] + [3] * 12
        base = pd.concat([base, extra], ignore_index=True)
        ev = blank(4)
        ev['geo_level_1_id'] = 1
        ev['geo_level_2_id'] = [100, 100, 100, 999]
        ev['geo_level_3_id'] = [8000001, 8000002, 123456789, 999999]          # thin, thick, unseen, unseen
        feats_b, (feats_e,) = F.make_features(base, [ev])
        all_high, all_low = float((base[F.GRADE] == 3).mean()), float((base[F.GRADE] == 1).mean())
        self.assertAlmostEqual(feats_e.loc[0, 'g3_high_rate'], all_high, places=12)     # thin key -> base-wide share
        self.assertAlmostEqual(feats_e.loc[0, 'g3_low_rate'], all_low, places=12)
        self.assertAlmostEqual(feats_e.loc[1, 'g3_high_rate'], 1.0, places=12)          # 12 rows all grade 3
        self.assertAlmostEqual(feats_e.loc[1, 'g3_low_rate'], 0.0, places=12)
        for i in (2, 3):                                                                  # unseen key -> base-wide share
            self.assertAlmostEqual(feats_e.loc[i, 'g3_high_rate'], all_high, places=12)
            self.assertAlmostEqual(feats_e.loc[i, 'g3_low_rate'], all_low, places=12)
        # k_material rate for eval rows: full-base share per material when count >= 50
        bf = F.building_features(base)
        mat = F.building_features(ev)['material_group']
        for i, m in enumerate(mat):
            sel = bf['material_group'] == m
            if sel.sum() >= 50:
                want_high, want_low = float((bf.loc[sel, F.GRADE] == 3).mean()), float((bf.loc[sel, F.GRADE] == 1).mean())
            else:
                want_high, want_low = all_high, all_low
            self.assertAlmostEqual(feats_e.loc[i, 'material_high_rate'], want_high, places=12)
            self.assertAlmostEqual(feats_e.loc[i, 'material_low_rate'], want_low, places=12)

    def test_gap_columns_are_rate_differences(self):
        base = synth(500, seed=22)
        ev = synth(60, seed=23, with_label=False)
        feats_b, (feats_e,) = F.make_features(base, [ev])
        for d in (feats_b, feats_e):
            for lv in ('g2', 'g3'):
                for h in ('high', 'low'):
                    np.testing.assert_allclose(d[f'{lv}_material_{h}_gap'].to_numpy(),
                                               (d[f'{lv}_material_{h}_rate'] - d[f'material_{h}_rate']).to_numpy(),
                                               rtol=0, atol=1e-15)

    def test_base_rate_is_out_of_fold(self):
        base = synth(500, seed=24)
        parts = list(KFold(5, shuffle=True, random_state=F.SEED).split(base))
        bf = F.building_features(base)
        fold_of = np.empty(len(base), dtype=int)
        for f, (_, this) in enumerate(parts):
            fold_of[this] = f
        # a row whose every combination key also occurs in its out-of-fold part (so no key falls back to the global share)
        row = None
        for i in range(len(base)):
            other = parts[fold_of[i]][0]
            if all((bf.loc[other, k] == bf.loc[i, k]).any() for k in F.RATE_KEYS):
                row = i
                break
        self.assertIsNotNone(row)
        before, _ = F.make_features(base, [])
        flipped = base.copy()
        flipped.loc[row, F.GRADE] = 1 if base.loc[row, F.GRADE] != 1 else 3
        after, _ = F.make_features(flipped, [])
        rate_cols = [c for c in F.feature_columns() if c.endswith('_rate')]
        pd.testing.assert_series_equal(before.loc[row, rate_cols], after.loc[row, rate_cols], check_exact=True)
        # a row in a different fold that shares a key does see the changed label (the check has teeth)
        changed = (before[rate_cols] != after[rate_cols]).any(axis=1)
        self.assertGreater(int(changed.sum()), 0)
        self.assertFalse(bool(changed.iloc[row]))

    def test_base_rate_matches_slow_independent_loop_for_one_key(self):
        base = synth(400, seed=25)
        bf = F.building_features(base)
        grade = base[F.GRADE].to_numpy()
        keys = bf['k_g2'].to_numpy()
        expect = np.zeros(len(base))
        all_high = (grade == 3).mean()
        for other, this in KFold(5, shuffle=True, random_state=F.SEED).split(base):
            other_high = (grade[other] == 3).mean()
            for i in this:
                same = [j for j in other if keys[j] == keys[i]]
                if not same:
                    expect[i] = all_high
                elif len(same) < F.RATE_KEYS['k_g2']:
                    expect[i] = other_high
                else:
                    expect[i] = np.mean(grade[same] == 3)
        feats_b, _ = F.make_features(base, [])
        np.testing.assert_allclose(feats_b['g2_high_rate'].to_numpy(), expect, rtol=0, atol=1e-15)


class EvalLabelInvarianceTests(unittest.TestCase):
    def setUp(self):
        fr = synth(500, seed=31)
        self.train = fr.iloc[:400].reset_index(drop=True)
        self.valid = fr.iloc[400:].reset_index(drop=True)

    def test_valid_matrix_identical_with_or_without_shuffled_labels(self):
        shuffled = self.valid.copy()
        shuffled[F.GRADE] = np.random.default_rng(1).permutation(shuffled[F.GRADE].to_numpy())
        self.assertFalse((shuffled[F.GRADE] == self.valid[F.GRADE]).all())
        _, a, _, _, _ = F.fold_matrices(self.train, self.valid.drop(columns=[F.GRADE]))
        _, b, _, _, _ = F.fold_matrices(self.train, self.valid)
        _, c, _, _, _ = F.fold_matrices(self.train, shuffled)
        self.assertTrue(a.equals(b))
        self.assertTrue(a.equals(c))
        self.assertEqual(F.matrix_sha256(a), F.matrix_sha256(c))

    def test_training_labels_do_change_the_valid_matrix(self):
        flipped = self.train.copy()
        flipped[F.GRADE] = (flipped[F.GRADE] % 3) + 1
        _, a, _, _, _ = F.fold_matrices(self.train, self.valid)
        _, b, _, _, _ = F.fold_matrices(flipped, self.valid)
        self.assertNotEqual(F.matrix_sha256(a), F.matrix_sha256(b))

    def test_label_probe_passes(self):
        res = F.label_probe(self.train, self.valid)
        self.assertTrue(res['passes'])
        self.assertTrue(res['valid_matrix_identical'])
        self.assertGreater(res['outer_labels_changed_share'], 0.2)

    def test_fold_matrices_shapes_and_diagnostics(self):
        valid = self.valid.copy()
        valid.loc[0, ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']] = [77, 7700, 770000]
        X_tr, X_va, f_tr, f_va, diag = F.fold_matrices(self.train, valid)
        self.assertEqual(list(X_tr.columns), list(X_va.columns))
        self.assertEqual(list(f_tr.columns), F.feature_columns())
        self.assertEqual(list(f_va.columns), F.feature_columns())
        self.assertEqual(len(X_tr), 400)
        self.assertEqual(len(X_va), 100)
        self.assertEqual(diag['columns'], X_tr.shape[1])
        self.assertEqual(diag['valid_rows_unseen_region'][1], 1)
        self.assertEqual(diag['nan_cells_train'], 0)
        self.assertEqual(diag['nan_cells_valid'], 0)
        self.assertEqual(X_va.loc[0, 'g1_n_buildings'], 0)


class ToMatrixTests(unittest.TestCase):
    def test_dummy_columns_come_from_the_first_frame(self):
        cols = ['age', 'position', 'material_group']
        tr = pd.DataFrame({'age': [1, 2, 3], 'position': ['o', 's', 'o'], 'material_group': ['RC', 'other', 'RC']})
        ev = pd.DataFrame({'age': [4, 5], 'position': ['t', 's'], 'material_group': ['cement', 'RC']})
        X_tr, X_ev = F.to_matrix([tr, ev], cols)
        self.assertEqual(list(X_tr.columns), ['age', 'position_o', 'position_s', 'material_group_RC', 'material_group_other'])
        self.assertEqual(list(X_ev.columns), list(X_tr.columns))
        self.assertEqual(X_ev.loc[0, ['position_o', 'position_s']].tolist(), [0, 0])         # 't' unseen -> all zero
        self.assertEqual(X_ev.loc[1, ['position_o', 'position_s']].tolist(), [0, 1])
        self.assertEqual(X_ev.loc[0, ['material_group_RC', 'material_group_other']].tolist(), [0, 0])  # 'cement' unseen
        self.assertEqual(X_ev.loc[1, 'material_group_RC'], 1)
        self.assertFalse(any('position_t' in c or 'cement' in c for c in X_ev.columns))
        self.assertEqual(X_ev['age'].tolist(), [4, 5])

    def test_category_missing_from_eval_is_filled_with_zero(self):
        cols = ['position']
        tr = pd.DataFrame({'position': ['o', 's', 't']})
        ev = pd.DataFrame({'position': ['s', 's']})
        _, X_ev = F.to_matrix([tr, ev], cols)
        self.assertEqual(list(X_ev.columns), ['position_o', 'position_s', 'position_t'])
        self.assertEqual(X_ev.to_numpy().tolist(), [[0, 1, 0], [0, 1, 0]])

    def test_non_dummy_columns_pass_through_and_input_is_untouched(self):
        cols = ['age', 'material_group']
        tr = pd.DataFrame({'age': [1.5, 2.5], 'material_group': ['RC', 'other'], 'extra': [9, 9]})
        before = tr.copy()
        (X,) = F.to_matrix([tr], cols)
        self.assertNotIn('extra', X.columns)
        pd.testing.assert_frame_equal(tr, before)


class ClassWeightTests(unittest.TestCase):
    def test_strength_half_and_zero(self):
        y0 = pd.Series([0] * 60 + [1] * 30 + [2] * 10)
        w = F.class_weights(y0, 0.5)
        self.assertEqual(len(w), 100)
        self.assertAlmostEqual(w[0], (100 / (3 * 60)) ** 0.5)
        self.assertAlmostEqual(w[60], (100 / (3 * 30)) ** 0.5)
        self.assertAlmostEqual(w[-1], (100 / (3 * 10)) ** 0.5)
        w0 = F.class_weights(y0, 0)
        np.testing.assert_array_equal(w0, np.ones(100))

    def test_weights_follow_row_labels_not_sort_order(self):
        y0 = pd.Series([2, 0, 0, 1, 0, 2])
        w = F.class_weights(y0, 1.0)
        self.assertAlmostEqual(w[0], 6 / (3 * 2))
        self.assertAlmostEqual(w[1], 6 / (3 * 3))
        self.assertAlmostEqual(w[3], 6 / (3 * 1))


class FingerprintTests(unittest.TestCase):
    def test_sha_changes_with_value_and_column_order_and_is_stable(self):
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [0.5, 0.25, 0.125]})
        h = F.matrix_sha256(df)
        self.assertEqual(h, F.matrix_sha256(df.copy()))
        self.assertEqual(len(h), 64)
        changed = df.copy()
        changed.loc[1, 'b'] += 1e-12
        self.assertNotEqual(h, F.matrix_sha256(changed))
        self.assertNotEqual(h, F.matrix_sha256(df[['b', 'a']]))
        self.assertNotEqual(h, F.matrix_sha256(df.rename(columns={'a': 'c'})))
        self.assertNotEqual(h, F.matrix_sha256(df.iloc[:2]))


@unittest.skipUnless(HAVE_PRIVATE, 'private notebook copy not present (kept outside Git)')
class OriginalFunctionTests(unittest.TestCase):
    """Bit-identity of our port against the notebook's own definitions (AST extract of the private copy)."""

    @classmethod
    def setUpClass(cls):
        if sha256_file(NOTEBOOK) != NOTEBOOK_SHA256:
            raise unittest.SkipTest('private notebook copy SHA differs from the registered one')
        cls.ns, cls.evidence = OC.extract(OC.cell_text(NOTEBOOK))

    def test_extract_has_functions_and_assignments(self):
        self.assertEqual(set(self.evidence['functions']), set(OC.NEEDED_FUNCTIONS))
        self.assertEqual(set(self.evidence['assignments']), set(OC.NEEDED_ASSIGNS))
        self.assertEqual(list(self.ns['ALL_FEATURES']), F.feature_columns())
        self.assertEqual(len(self.evidence['extract_text_sha256']), 64)
        for name in OC.NEEDED_FUNCTIONS:
            self.assertTrue(callable(self.ns[name]))

    def test_compare_is_bit_identical_on_synthetic_fold(self):
        fr = synth(700, seed=41, regions=5)
        train = fr.iloc[:560].reset_index(drop=True)
        valid = fr.iloc[560:].reset_index(drop=True)
        valid.loc[0, ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']] = [77, 7700, 770000]   # unseen regions
        self.assertTrue((train['geo_level_1_id'].value_counts() >= F.MIN_REGION).all())
        self.assertTrue((train['geo_level_3_id'].value_counts() < F.MIN_REGION).any())
        res = OC.compare(self.ns, train, valid)
        self.assertTrue(res['passes'], res)
        self.assertEqual(set(res['checks']), {'features_train', 'features_valid', 'matrix_train', 'matrix_valid', 'weights'})
        self.assertEqual(res['matrix_columns'], F.fold_matrices(train, valid.drop(columns=[F.GRADE], errors='ignore'))[0].shape[1])

    def test_compare_detects_a_deviation(self):
        fr = synth(400, seed=42, regions=4)
        train, valid = fr.iloc[:320].reset_index(drop=True), fr.iloc[320:].reset_index(drop=True)
        other = OC.compare(self.ns, train, valid)
        self.assertTrue(other['passes'], other)
        ok, why = OC.frames_identical(pd.DataFrame({'x': [1.0, 2.0]}), pd.DataFrame({'x': [1.0, 2.0 + 1e-15]}))
        self.assertFalse(ok)
        self.assertTrue(why)


class RunnerContractTests(unittest.TestCase):
    """Runner policy without data: frozen config, budget refusal, unit file names, rule labels, holdout guard."""

    @classmethod
    def setUpClass(cls):
        from experiments.ey_pipeline_v2 import run as R
        cls.R = R

    def test_config_is_the_registered_plan(self):
        c = self.R.config()
        self.assertEqual(c['budget']['planned_fits'], 10)
        self.assertEqual(c['budget']['max_fits'], 12)
        self.assertEqual(c['weights']['strength'], 0.5)
        self.assertEqual(c['lgbm_params']['n_estimators'], 800)
        self.assertEqual(c['xgb_params']['n_estimators'], 800)
        self.assertEqual(self.R.targets()[:2], [('L', 0), ('X', 0)])
        self.assertEqual(len(self.R.targets()), 10)

    def test_changed_config_is_refused(self):
        import json
        import tempfile
        c = json.loads((ROOT/'experiments'/'ey_pipeline_v2'/'config.json').read_text(encoding='utf-8'))
        c['lgbm_params']['num_leaves'] = 63
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'config.json'
            p.write_text(json.dumps(c), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.R.config(p)

    def test_budget_counts_and_unit_files(self):
        ledger = dict(units={'L/0': dict(attempts=1), 'X/0': dict(attempts=2), 'L/1': dict(attempts=1)})
        self.assertEqual(self.R.fits_started(ledger), 4)
        self.assertEqual(self.R.recovered_units(ledger), 1)
        self.assertEqual(self.R.unit_files('L', 3), ['fold_3_model.txt', 'fold_3_pred.csv.gz', 'fold_3.json'])
        self.assertEqual(self.R.unit_files('X', 0), ['fold_0_model.json', 'fold_0_pred.csv.gz', 'fold_0.json'])

    def test_d3_rule_labels(self):
        rule = self.R.config()['rules']['D3']
        from experiments.ey_pipeline_benchmark.run import d1_label
        self.assertEqual(d1_label([.71] * 5, [.70] * 5, rule)['label'], 'LGBM_AHEAD_DESCRIPTIVE')
        self.assertEqual(d1_label([.70] * 5, [.71] * 5, rule)['label'], 'XGB_AHEAD_DESCRIPTIVE')
        self.assertEqual(d1_label([.7015] * 5, [.70] * 5, rule)['label'], 'SIMILAR')

    def test_holdout_path_refused(self):
        with self.assertRaises(ValueError):
            self.R.require_not_holdout('C:/x/earthquake/data/final_holdout/test.csv')


if __name__ == '__main__':
    unittest.main()
