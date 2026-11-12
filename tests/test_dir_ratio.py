"""
Tests for diper.dir_ratio (directionality ratio d/D over elapsed time).
"""
import unittest

import numpy as np
import pandas as pd

from diper.dir_ratio import align_and_average_dir_ratios, calculate_directionality_ratio


def _track(frames, xs, ys):
    return pd.DataFrame({'frame': frames, 'x': xs, 'y': ys})


class TestDirRatio(unittest.TestCase):

    def test_straight_line_ratio_is_one(self):
        res = calculate_directionality_ratio(_track([1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 0, 0]), 2.0)
        np.testing.assert_allclose(res['dir_ratio'], 1.0)

    def test_time_is_elapsed_time_from_track_start(self):
        """Time counts from the first point of each track (0, dt, 2dt...), whatever its frame numbers."""
        res = calculate_directionality_ratio(_track([37, 38, 39], [0, 1, 1], [0, 0, 1]), 0.5)
        np.testing.assert_allclose(res['time'], [0.0, 0.5, 1.0])

    def test_tracks_starting_at_different_frames_are_aligned(self):
        """Two tracks with the same shape but different start frames give one averaged curve, n = 2 at each time."""
        a = calculate_directionality_ratio(_track([1, 2, 3], [0, 1, 1], [0, 0, 1]), 1.0)
        b = calculate_directionality_ratio(_track([20, 21, 22], [5, 6, 6], [5, 5, 6]), 1.0)
        avg = align_and_average_dir_ratios([a, b])
        self.assertEqual(list(avg['n']), [2, 2, 2])
        np.testing.assert_allclose(avg['avg_dir_ratio'], [1.0, 1.0, np.sqrt(2) / 2])

    def test_sparsed_track_uses_time_interval_per_row(self):
        """After Sparse_Data (frames 1, 4, 7 kept, time_interval multiplied by 3) rows are dt apart."""
        res = calculate_directionality_ratio(_track([1, 4, 7], [0, 3, 6], [0, 0, 0]), 3.0)
        np.testing.assert_allclose(res['time'], [0.0, 3.0, 6.0])


if __name__ == '__main__':
    unittest.main()
