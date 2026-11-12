"""
Error bars are the sample standard error of the mean: std with ddof=1, divided by sqrt(n).
"""
import numpy as np
import pandas as pd
import pytest

from diper.autocorrel import average_autocorrelation_results
from diper.autocorrel_3d import average_autocorrelation_results as average_3d
from diper.autocorrel_nogaps import average_autocorrelation_results as average_nogaps
from diper.dir_ratio import align_and_average_dir_ratios
from diper.msd import average_msd_results
from diper.utils import sem
from diper.vel_cor import average_vel_autocorr_results

VALUES = [0.2, 0.5, 0.9]
EXPECTED = np.std(VALUES, ddof=1) / np.sqrt(3)


def test_sem_helper():
    assert sem(VALUES) == pytest.approx(EXPECTED, rel=1e-15)
    assert sem(VALUES) == pytest.approx(pd.Series(VALUES).sem(), rel=1e-15)
    assert np.isnan(sem([0.3]))


@pytest.mark.parametrize("fn,col", [(average_msd_results, "msd"), (average_vel_autocorr_results, "vel_autocorr")])
def test_curve_averages_use_sample_sem(fn, col):
    out = fn([pd.DataFrame({"time_interval": [1.0], col: [v]}) for v in VALUES])
    assert out["sem"].iloc[0] == pytest.approx(EXPECTED, rel=1e-12)


@pytest.mark.parametrize("fn", [average_autocorrelation_results, average_3d, average_nogaps])
def test_autocorrelation_averages_use_sample_sem(fn):
    out = fn([pd.DataFrame({"time_interval": [1.0], "autocorrelation": [v]}) for v in VALUES])
    assert out["sem"].iloc[0] == pytest.approx(EXPECTED, rel=1e-12)


def test_dir_ratio_average_uses_sample_sem():
    out = align_and_average_dir_ratios([pd.DataFrame({"time": [0.0], "dir_ratio": [v]}) for v in VALUES])
    assert out["sem"].iloc[0] == pytest.approx(EXPECTED, rel=1e-12)
