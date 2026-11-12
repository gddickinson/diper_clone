"""
Population averaging of direction / velocity autocorrelation (Gorelik & Gautreau 2014, eq 5):
<DA> = sum_j DA_j * N_j / sum_k N_k, with N_j the number of displacements of cell j. The unweighted mean of the
per-cell curves stays available as weighting="tracks".
"""
import numpy as np
import pandas as pd
import pytest

from diper.autocorrel import run_autocorrelation_analysis
from diper.autocorrel_3d import run_autocorrelation_3d_analysis
from diper.autocorrel_nogaps import run_autocorrelation_nogaps_analysis
from diper.utils import average_curves
from diper.vel_cor import run_vel_cor_analysis


def _data(z=False):
    """Cell 1: straight, 11 points = 10 displacements, DA(1) = 1. Cell 2: there and back, 3 points = 2
    displacements, DA(1) = -1. Weighted DA(1) = (10 - 2) / 12; unweighted = 0."""
    frames = list(range(1, 12)) + [1, 2, 3]
    x = [float(i) for i in range(11)] + [0.0, 1.0, 0.0]
    df = pd.DataFrame({"frame": frames, "x": x, "y": 0.0})
    if z:
        df["z"] = 0.0
    return {"c": df}


RUNS = [
    (lambda w, out: run_autocorrelation_analysis(_data(), 1.0, 5, out, weighting=w)["c"], "avg_autocorr"),
    (lambda w, out: run_autocorrelation_3d_analysis(_data(z=True), 1.0, 5, out, weighting=w)["c"], "avg_autocorr"),
    (lambda w, out: run_autocorrelation_nogaps_analysis(_data(), 1.0, 5, 0.0, out, weighting=w)["c"], "avg_autocorr"),
    (lambda w, out: run_vel_cor_analysis(_data(), 1.0, output_dir=out, weighting=w)["c"], "avg_vel_autocorr"),
]


@pytest.mark.parametrize("run,col", RUNS, ids=["autocorrel", "autocorrel_3d", "autocorrel_nogaps", "vel_cor"])
def test_weighted_by_displacements_by_default(run, col, tmp_path):
    res = run("displacements", str(tmp_path))
    assert res[col].iloc[0] == pytest.approx(8 / 12, rel=1e-12)
    assert int(res["n"].iloc[0]) == 2
    assert res["sem"].iloc[0] == pytest.approx(np.std([1, -1], ddof=1) / np.sqrt(2), rel=1e-12)


@pytest.mark.parametrize("run,col", RUNS, ids=["autocorrel", "autocorrel_3d", "autocorrel_nogaps", "vel_cor"])
def test_unweighted_per_track_option(run, col, tmp_path):
    res = run("tracks", str(tmp_path))
    assert res[col].iloc[0] == pytest.approx(0.0, abs=1e-15)


def test_default_is_paper_weighting(tmp_path):
    a = run_autocorrelation_analysis(_data(), 1.0, 5, str(tmp_path))["c"]
    assert a["avg_autocorr"].iloc[0] == pytest.approx(8 / 12)


def test_average_curves_helper():
    curves = [pd.DataFrame({"t": [1.0, 2.0], "v": [1.0, 0.5]}), pd.DataFrame({"t": [1.0], "v": [-1.0]})]
    out = average_curves(curves, "t", "v", weights=[3, 1])
    assert list(out["avg"]) == [pytest.approx(0.5), pytest.approx(0.5)]
    assert list(out["n"]) == [2, 1]
    with pytest.raises(ValueError):
        average_curves(curves, "t", "v", weights=[1])
