"""T23 확률 변환의 손계산·투영·유효성 검사. 분류기를 학습하지 않는다."""
import json
import unittest

import numpy as np

from experiments.ordinal_damage.model import damage_argmax, project_cumulative_probabilities


class OrdinalDamageModelTest(unittest.TestCase):
    def test_noncrossing_hand_values_and_label_mapping(self):
        p = project_cumulative_probabilities([0., 1., 1., .75], [0., 0., 1., .25])
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.25, .5, .25]], dtype=np.float64)
        np.testing.assert_array_equal(p, expected)
        np.testing.assert_array_equal(damage_argmax(p), [1, 2, 3, 2])

    def test_crossing_projects_both_to_mean_and_middle_mass_zero(self):
        p, m = project_cumulative_probabilities([.25, .75, 0], [.75, .25, 1], return_metadata=True)
        np.testing.assert_array_equal(p, [[.5, 0, .5], [.25, .5, .25], [.5, 0, .5]])
        self.assertEqual(m, dict(rows=3, crossing_count=2, crossing_rate=2/3,
            crossing_excess_sum=1.5, mean_crossing_excess=.75,
            mean_crossing_excess_denominator=2, max_crossing_excess=1.))
        np.testing.assert_array_equal(damage_argmax(p), [1, 2, 1])

    def test_exact_ties_choose_first_of_labels_123(self):
        p = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1/3, 1/3, 1/3]])
        np.testing.assert_array_equal(damage_argmax(p), [1, 2, 1, 1])
        near = np.array([[.5, np.nextafter(.5, 1), 0]])
        np.testing.assert_array_equal(damage_argmax(near), [2])

    def test_crossing_check_is_strict_without_epsilon(self):
        q2 = np.array([.5, .5, np.nextafter(.5, 1)])
        q3 = np.array([np.nextafter(.5, 1), .5, .5])
        p, m = project_cumulative_probabilities(q2, q3, return_metadata=True)
        self.assertEqual(m['crossing_count'], 1)
        self.assertEqual(m['mean_crossing_excess'], q3[0]-q2[0])
        self.assertEqual(p[0, 1], 0)
        self.assertGreater(p[2, 1], 0)

    def test_float32_inputs_use_float64_arithmetic_without_changing_input(self):
        q2 = np.array([.1, .7, .9], dtype=np.float32)
        q3 = np.array([.2, .4, .95], dtype=np.float32)
        before2, before3 = q2.copy(), q3.copy()
        p = project_cumulative_probabilities(q2, q3)
        expected = []
        for a, b in zip(q2.tolist(), q3.tolist()):
            if b > a:
                a = b = (a+b)/2
            expected.append([1-a, a-b, b])
        self.assertEqual(p.dtype, np.float64)
        np.testing.assert_array_equal(p, expected)
        np.testing.assert_array_equal(q2, before2)
        np.testing.assert_array_equal(q3, before3)

    def test_readonly_and_shared_input_views_are_supported(self):
        base = np.array([.25, .75, .5, .5])
        base.flags.writeable = False
        p = project_cumulative_probabilities(base[:2], base[2:])
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(base, [.25, .75, .5, .5])
        identity = project_cumulative_probabilities(base, base)
        np.testing.assert_array_equal(identity[:, 1], np.zeros(4))

    def test_shapes_reject_scalar_column_matrix_and_length_mismatch(self):
        for a, b in [(.5, .2), ([[.5]], [.2]), ([.5], [[.2]]),
                     (np.ones((2, 2)), [.2, .3]), ([.5, .6], [.2])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                project_cumulative_probabilities(a, b)

    def test_nonfinite_range_and_nonreal_inputs_rejected(self):
        for bad in [[np.nan], [np.inf], [-np.inf], [-.01], [1.01],
                    [np.nextafter(0., -1)], [np.nextafter(1., 2)],
                    ['.5'], [None], [complex(.5, .1)], [True]]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    project_cumulative_probabilities(bad, [.2])
                with self.assertRaises(ValueError):
                    project_cumulative_probabilities([.8], bad)

    def test_empty_and_noncrossing_metadata_denominators(self):
        p, m = project_cumulative_probabilities([], [], return_metadata=True)
        self.assertEqual(p.shape, (0, 3))
        self.assertEqual(p.dtype, np.float64)
        self.assertEqual(m, dict(rows=0, crossing_count=0, crossing_rate=None,
            crossing_excess_sum=0., mean_crossing_excess=0.,
            mean_crossing_excess_denominator=0, max_crossing_excess=0.))
        self.assertEqual(damage_argmax(p).shape, (0,))
        _, m = project_cumulative_probabilities([.8, .6], [.2, .6], return_metadata=True)
        self.assertEqual(m['crossing_rate'], 0.)
        self.assertEqual(m['mean_crossing_excess_denominator'], 0)
        self.assertEqual(json.loads(json.dumps(m, allow_nan=False)), m)

    def test_random_probability_simplex_monotonicity_and_independent_loop(self):
        rng = np.random.default_rng(23)
        q2, q3 = rng.random(1007), rng.random(1007)
        p, m = project_cumulative_probabilities(q2, q3, return_metadata=True)
        expected = []
        for a, b in zip(q2.tolist(), q3.tolist()):
            if b > a:
                a = b = (a+b)/2.
            expected.append([1.-a, a-b, b])
        np.testing.assert_array_equal(p, expected)
        self.assertTrue(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15, rtol=0)
        self.assertEqual(m['crossing_count'], int(np.count_nonzero(q3 > q2)))
        self.assertTrue(np.all(p[:, 1]+p[:, 2] >= p[:, 2]))
        again, second = project_cumulative_probabilities(p[:, 1]+p[:, 2], p[:, 2], return_metadata=True)
        np.testing.assert_allclose(again, p, atol=2e-16, rtol=0)
        self.assertEqual(second['crossing_count'], 0)

    def test_crossing_projection_minimizes_equal_weight_squared_distance(self):
        p = project_cumulative_probabilities([.2], [.8])
        projected = np.array([p[0, 1]+p[0, 2], p[0, 2]])
        np.testing.assert_array_equal(projected, [.5, .5])
        projected_distance = np.square(projected-[.2, .8]).sum()
        grid = np.linspace(0, 1, 101)
        allowed = np.array([[a, b] for a in grid for b in grid if b <= a])
        distances = np.square(allowed-[.2, .8]).sum(axis=1)
        self.assertLessEqual(projected_distance, float(distances.min())+1e-15)

    def test_argmax_rejects_invalid_probability_matrices(self):
        for bad in [[.5, .5, 0], [[.5, .5]], [[.5, .5, 0, 0]],
                    [[np.nan, 0, 1]], [[-.1, .1, 1]], [[1.1, 0, 0]],
                    [[.2, .2, .2]], [['.5', '.5', '0']], [[.5+0j, .5, 0]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                damage_argmax(bad)

    def test_float32_argmax_accepts_representation_roundoff_without_renormalizing(self):
        p = np.array([[.1, .2, .7], [.45, .45, .1]], dtype=np.float32)
        before = p.copy()
        np.testing.assert_array_equal(damage_argmax(p), [3, 1])
        np.testing.assert_array_equal(p, before)


if __name__ == '__main__':
    unittest.main()
