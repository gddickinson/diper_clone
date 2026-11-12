"""
Tests for diper.vel_cor (normalized velocity autocorrelation, Gorelik & Gautreau 2014, eqs 6-7).
"""
import unittest

import numpy as np
import pandas as pd

from diper.vel_cor import calculate_normalized_velocity_autocorrelation


class TestVelocityAutocorrelation(unittest.TestCase):

    def test_constant_velocity_is_one(self):
        """Straight movement at constant speed: v(t).v(t+n) = <v^2>, so every coefficient is 1."""
        n = 30
        traj = pd.DataFrame({'x': np.arange(n) * 2.0, 'y': np.arange(n) * 1.0})
        res = calculate_normalized_velocity_autocorrelation(traj, time_interval=0.5)
        self.assertGreater(len(res), 2)
        np.testing.assert_allclose(res['vel_autocorr'].to_numpy(dtype=float), 1.0, rtol=1e-12)

    def test_norm_is_mean_square_velocity(self):
        """Norm (eq 6) is the MEAN squared velocity, so the result does not shrink with track length."""
        rng = np.random.default_rng(1)
        steps = rng.normal(size=(40, 2)) + [3.0, 0.0]
        xy = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
        traj = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1]})
        res = calculate_normalized_velocity_autocorrelation(traj, time_interval=1.0)
        v = steps
        expected = np.mean(np.sum(v[:-1] * v[1:], axis=1)) / np.mean(np.sum(v ** 2, axis=1))
        self.assertAlmostEqual(float(res['vel_autocorr'].iloc[0]), expected, places=12)


if __name__ == '__main__':
    unittest.main()
