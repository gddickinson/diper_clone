"""
Tests for diper.utils module.
"""
import os
import unittest
import tempfile
import shutil

import numpy as np
import pandas as pd

from diper.utils import split_trajectories, ensure_output_dir, save_figure, save_results


class TestSplitTrajectories(unittest.TestCase):
    """Tests for trajectory splitting logic."""

    def test_single_trajectory(self):
        """A single trajectory with increasing frames should return one trajectory."""
        df = pd.DataFrame({
            'frame': [1, 2, 3, 4, 5],
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        result = split_trajectories(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 5)

    def test_two_trajectories(self):
        """Frame reset should split into two trajectories."""
        df = pd.DataFrame({
            'frame': [1, 2, 3, 1, 2, 3],
            'x': [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            'y': [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
        })
        result = split_trajectories(df)
        self.assertEqual(len(result), 2)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame({'frame': [], 'x': [], 'y': []})
        result = split_trajectories(df)
        self.assertEqual(len(result), 0)

    def test_single_row(self):
        """Single row returns the whole df as a trajectory (fallback behavior)."""
        df = pd.DataFrame({
            'frame': [1],
            'x': [0.0],
            'y': [0.0],
        })
        result = split_trajectories(df)
        # split_trajectories returns [df] for single-row input
        self.assertEqual(len(result), 1)

    def test_same_frame_splits(self):
        """Same frame number repeated should trigger a split."""
        df = pd.DataFrame({
            'frame': [1, 2, 2, 3, 4],
            'x': [0.0, 1.0, 5.0, 6.0, 7.0],
            'y': [0.0, 0.0, 5.0, 5.0, 5.0],
        })
        result = split_trajectories(df)
        self.assertGreaterEqual(len(result), 2)


class TestEnsureOutputDir(unittest.TestCase):
    """Tests for directory creation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_creates_directory(self):
        target = os.path.join(self.test_dir, "new_output")
        result = ensure_output_dir(target)
        self.assertTrue(os.path.isdir(result))

    def test_creates_subdirectory(self):
        result = ensure_output_dir(self.test_dir, "sub")
        expected = os.path.join(self.test_dir, "sub")
        self.assertEqual(result, expected)
        self.assertTrue(os.path.isdir(expected))

    def test_existing_directory_no_error(self):
        """Should not raise if directory already exists."""
        result = ensure_output_dir(self.test_dir)
        self.assertTrue(os.path.isdir(result))


class TestSaveResults(unittest.TestCase):
    """Tests for saving results to files."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_csv(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        save_results(df, self.test_dir, "test_output")
        csv_path = os.path.join(self.test_dir, "test_output.csv")
        self.assertTrue(os.path.exists(csv_path))
        loaded = pd.read_csv(csv_path)
        self.assertEqual(len(loaded), 2)


if __name__ == '__main__':
    unittest.main()
