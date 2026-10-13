"""Synthetic-only model checks. These are not real-data performance measurements."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline import MODEL_NAMES, aligned_probabilities, build_pipeline, evaluate_model, metrics, run, select_positions
from preprocess import ALLOWED_TEXT, BINARY, FEATURES, GEO, NUMERIC, normalize_features


def synthetic(n=150):
    rng = np.random.default_rng(42)
    values = {c: rng.integers(0, 8, n) for c in GEO}
    values.update(dict(zip(NUMERIC, [rng.integers(1, 4, n), rng.integers(0, 50, n),
                                     rng.integers(1, 20, n), rng.integers(1, 15, n), rng.integers(0, 4, n)])))
    values.update({c: rng.integers(0, 2, n) for c in BINARY})
    values.update({c: rng.choice(allowed, n) for c, allowed in ALLOWED_TEXT.items()})
    return normalize_features(pd.DataFrame(values)[FEATURES]), pd.Series(np.arange(n) % 3 + 1), pd.Series(np.arange(1, n + 1)), np.arange(n) % 5


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/baseline.json').read_text(encoding='utf-8'))

    def test_metrics_match_hand_calculation(self):
        result = metrics([1, 1, 2, 2, 3, 3], [1, 2, 2, 2, 3, 1])
        self.assertAlmostEqual(result['macro_f1'], (0.5 + 0.8 + 2 / 3) / 3)
        self.assertAlmostEqual(result['severe_as_minor_rate'], 0.5)
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_invalid_labels_and_shapes_rejected(self):
        for true, pred in [([1], [0]), ([4], [1]), ([1, 2], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                metrics(true, pred)

    def test_class_probabilities_reordered(self):
        class Model:
            classes_ = np.array([3, 1, 2])
        class FakePipeline:
            named_steps = {'model': Model()}
            def predict_proba(self, X):
                return np.array([[0.6, 0.1, 0.3]])
        np.testing.assert_allclose(aligned_probabilities(FakePipeline(), None), [[0.1, 0.3, 0.6]])

    def test_pilot_is_deterministic_and_retains_original_folds(self):
        _, y, _, folds = synthetic(600)
        a = select_positions(y, folds, 150, 42)
        np.testing.assert_array_equal(a, select_positions(y, folds, 150, 42))
        self.assertEqual(len(a), 150)
        self.assertEqual(set(folds[a]), set(range(5)))
        for fold in range(5):
            self.assertEqual(set(y.iloc[a[folds[a] == fold]]), {1, 2, 3})

    def test_lr_scaling_is_training_only_and_output_sparse(self):
        X, y, _, _ = synthetic()
        model = build_pipeline('lr', self.config, jobs=1).fit(X.iloc[:120], y.iloc[:120])
        scaler = model['preprocess']['columns'].named_transformers_['numeric']['scale']
        np.testing.assert_allclose(scaler.mean_, X.iloc[:120][NUMERIC].mean())
        before = scaler.mean_.copy()
        changed = X.iloc[120:].copy()
        changed['age'] = 995
        self.assertTrue(sparse.issparse(model['preprocess'].transform(changed)))
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertFalse(hasattr(clone(model)['model'], 'classes_'))

    def test_all_models_five_folds_oof_and_reload(self):
        X, y, ids, folds = synthetic()
        config = copy.deepcopy(self.config)
        config['models']['rf']['n_estimators'] = 3
        config['models']['gbc']['n_estimators'] = 3
        with tempfile.TemporaryDirectory() as temp:
            for name in MODEL_NAMES:
                with self.subTest(model=name):
                    result = evaluate_model(name, X, y, ids, folds, config, Path(temp) / name, jobs=1)
                    self.assertEqual(result['folds'], 5)
                    self.assertTrue(result['exactly_one_oof_per_row'])
                    self.assertFalse(result['holdout_scored'])
                    oof = pd.read_csv(Path(temp) / name / 'oof_predictions.csv.gz')
                    self.assertEqual(len(oof), len(X))
                    self.assertTrue(oof.building_id.is_unique)
                    self.assertTrue(result['fold_results'][0]['reload_prediction_checked'])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            build_pipeline('not-a-model', self.config)

    def test_nonempty_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'output'
            output.mkdir()
            (output / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                run('missing-data', output, ROOT / 'configs/baseline.json')
            self.assertEqual((output / 'keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
