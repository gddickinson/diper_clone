"""
Tests for diper.autocorrel module.
"""
import unittest
import numpy as np
import pandas as pd

from diper.autocorrel import (
    normalize_vectors,
    calculate_autocorrelation,
    average_autocorrelation_results,
)


class TestNormalizeVectors(unittest.TestCase):
    """Tests for vector normalization."""

    def test_straight_line_x(self):
        """A straight trajectory along x should give unit vectors (1, 0)."""
        traj = pd.DataFrame({
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        result = normalize_vectors(traj)
        # After normalization, last row is dropped, so 4 rows of vectors
        self.assertEqual(len(result), 4)
        # All x_vec should be ~1.0 (moving in positive x)
        np.testing.assert_array_almost_equal(result['x_vec'].values, [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result['y_vec'].values, [0.0, 0.0, 0.0, 0.0])

    def test_diagonal_trajectory(self):
        """A 45-degree trajectory should give normalized (0.707, 0.707) vectors."""
        traj = pd.DataFrame({
            'x': [0.0, 1.0, 2.0, 3.0],
            'y': [0.0, 1.0, 2.0, 3.0],
        })
        result = normalize_vectors(traj)
        expected = 1.0 / np.sqrt(2.0)
        np.testing.assert_array_almost_equal(result['x_vec'].values, [expected] * 3)
        np.testing.assert_array_almost_equal(result['y_vec'].values, [expected] * 3)

    def test_stationary_point(self):
        """Zero displacement should give zero vectors."""
        traj = pd.DataFrame({
            'x': [0.0, 0.0, 1.0],
            'y': [0.0, 0.0, 1.0],
        })
        result = normalize_vectors(traj)
        # First vector (0,0)->(0,0) should be 0, second should be non-zero
        self.assertEqual(result['x_vec'].iloc[0], 0)
        self.assertEqual(result['y_vec'].iloc[0], 0)

    def test_single_point(self):
        """Single point trajectory should return the input (no vectors to compute)."""
        traj = pd.DataFrame({'x': [0.0], 'y': [0.0]})
        result = normalize_vectors(traj)
        # normalize_vectors returns the input unchanged for length <= 1
        self.assertEqual(len(result), 1)


class TestCalculateAutocorrelation(unittest.TestCase):
    """Tests for autocorrelation computation."""

    def test_perfect_persistence(self):
        """A straight line should have autocorrelation close to 1.0 at all intervals."""
        traj = pd.DataFrame({
            'x': np.arange(20, dtype=float),
            'y': np.zeros(20),
        })
        traj_norm = normalize_vectors(traj)
        result = calculate_autocorrelation(traj_norm, time_interval=1.0, max_intervals=5)
        # All autocorrelations should be ~1.0 for a straight line
        self.assertGreater(len(result), 0)
        for val in result['autocorrelation']:
            self.assertAlmostEqual(val, 1.0, places=5)

    def test_short_trajectory(self):
        """Very short trajectory should return limited results."""
        traj = pd.DataFrame({
            'x': [0.0, 1.0],
            'y': [0.0, 0.0],
        })
        traj_norm = normalize_vectors(traj)
        result = calculate_autocorrelation(traj_norm, time_interval=1.0, max_intervals=10)
        # Only one point after normalization, so no autocorrelation possible
        self.assertEqual(len(result), 0)


class TestAverageAutocorrelation(unittest.TestCase):
    """Tests for averaging autocorrelation results."""

    def test_average_identical_results(self):
        """Averaging identical results should give the same values with zero SEM."""
        df1 = pd.DataFrame({
            'time_interval': [1.0, 2.0, 3.0],
            'autocorrelation': [0.8, 0.5, 0.2],
        })
        df2 = df1.copy()
        result = average_autocorrelation_results([df1, df2])
        np.testing.assert_array_almost_equal(result['avg_autocorr'].values, [0.8, 0.5, 0.2])
        # SEM should be 0 for identical values
        np.testing.assert_array_almost_equal(result['sem'].values, [0.0, 0.0, 0.0])

    def test_empty_list(self):
        """Empty list should return empty DataFrame."""
        result = average_autocorrelation_results([])
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()
