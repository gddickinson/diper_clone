"""
Tests for diper.msd module.
"""
import unittest
import numpy as np
import pandas as pd

from diper.msd import calculate_msd, average_msd_results


class TestCalculateMSD(unittest.TestCase):
    """Tests for MSD computation."""

    def test_straight_line(self):
        """MSD for a straight line at constant speed should be quadratic: MSD = (v*t)^2."""
        n = 50
        traj = pd.DataFrame({
            'x': np.arange(n, dtype=float),
            'y': np.zeros(n),
        })
        result = calculate_msd(traj, time_interval=1.0, max_interval_fraction=0.25)
        self.assertGreater(len(result), 0)
        # For a straight line moving 1 unit/step: MSD at step k should be k^2
        for _, row in result.iterrows():
            step = int(row['time_interval'])
            expected_msd = step ** 2
            self.assertAlmostEqual(row['msd'], expected_msd, places=5)

    def test_stationary(self):
        """MSD for a stationary trajectory should be 0."""
        n = 20
        traj = pd.DataFrame({
            'x': np.zeros(n),
            'y': np.zeros(n),
        })
        result = calculate_msd(traj, time_interval=1.0, max_interval_fraction=0.5)
        self.assertGreater(len(result), 0)
        for val in result['msd']:
            self.assertAlmostEqual(val, 0.0, places=10)

    def test_single_point(self):
        """Single point trajectory should return empty result."""
        traj = pd.DataFrame({'x': [0.0], 'y': [0.0]})
        result = calculate_msd(traj, time_interval=1.0)
        self.assertEqual(len(result), 0)


class TestAverageMSD(unittest.TestCase):
    """Tests for averaging MSD results."""

    def test_average_identical(self):
        """Averaging identical MSD results should reproduce the same values."""
        df = pd.DataFrame({
            'time_interval': [1.0, 2.0, 3.0],
            'msd': [1.0, 4.0, 9.0],
        })
        result = average_msd_results([df, df.copy()])
        np.testing.assert_array_almost_equal(result['avg_msd'].values, [1.0, 4.0, 9.0])

    def test_empty_list(self):
        """Empty list should return empty DataFrame."""
        result = average_msd_results([])
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()
