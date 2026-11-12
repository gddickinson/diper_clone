"""
Tests for diper.plot_at_origin.
"""
import unittest

import pandas as pd

from diper.plot_at_origin import determine_plot_area, translate_to_origin


class TestPlotArea(unittest.TestCase):

    def test_area_contains_every_trajectory(self):
        """All tracks start at (0, 0); tracks that all head one way must still fit inside the square area."""
        tracks = [translate_to_origin(pd.DataFrame({'x': [5, 50, 105], 'y': [5, 20, 30]})),
                  translate_to_origin(pd.DataFrame({'x': [0, 30], 'y': [0, -10]}))]
        xmin, xmax, ymin, ymax = determine_plot_area(tracks)
        for t in tracks:
            self.assertGreaterEqual(t['relative_x'].min(), xmin)
            self.assertLessEqual(t['relative_x'].max(), xmax)
            self.assertGreaterEqual(t['relative_y'].min(), ymin)
            self.assertLessEqual(t['relative_y'].max(), ymax)
        self.assertEqual(xmin, -xmax)
        self.assertEqual(xmax, ymax)

    def test_origin_centred_rounded_to_ten(self):
        t = translate_to_origin(pd.DataFrame({'x': [0, -42], 'y': [0, 7]}))
        self.assertEqual(determine_plot_area([t]), (-50.0, 50.0, -50.0, 50.0))


if __name__ == '__main__':
    unittest.main()
