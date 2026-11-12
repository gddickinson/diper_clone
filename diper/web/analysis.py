"""Bridge between the web API and the repo's real DiPer code: nothing here re-implements the maths.

Each uploaded CSV (one condition) or Excel sheet (one condition per sheet) is read with `diper.utils.load_data`, then:

  1. preparation (web layer only): frame/x/y (and z, column 7) are coerced to numbers, rows with a missing frame, x or y
     are dropped, and x/y/z are multiplied by the pixel size when the coordinates are in pixels, so every length below
     is in micrometres;
  2. optional Sparse_Data: `sparse_data.sparse_trajectory` keeps 1 of N rows per track and the time interval becomes
     N x dt (the manual's advice);
  3. tracks: `utils.split_trajectories`;
  4. the analyses, called exactly as the run_* functions call them (minus their file/plot output):
     Plot_At_Origin (`determine_plot_area`), Speed, Dir_Ratio, MSD (+ alpha), Autocorrel, Autocorrel_3D (when a 7th
     column is present), Vel_Cor.

The JSON shape is the one diper_engine.js (the JS port used on the static site) produces; the fixture test checks both.
"""
import contextlib
import io
import os
import re
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from diper import autocorrel, autocorrel_3d, dir_ratio, msd, plot_at_origin, speed, vel_cor  # noqa: E402
from diper.sparse_data import sparse_trajectory  # noqa: E402
from diper.utils import load_data, split_trajectories  # noqa: E402

MAX_SEGMENTS_RETURNED = 300  # tracks sent back for plotting (per condition); the analysis always uses all
DEFAULTS = {"time_interval": 1.0, "max_intervals": 30, "sparse_n": 1,
            "msd_max_interval_fraction": 0.5, "msd_alpha_fraction": 0.1, "vel_cor_max_step_fraction": 1 / 3}


def _num(v):
    """JSON-safe float: NaN/inf become None."""
    v = float(v)
    return v if np.isfinite(v) else None


def _nums(values):
    return [_num(v) for v in values]


def _curve(avg: pd.DataFrame, tcol: str, vcol: str) -> dict:
    return {"t": _nums(avg[tcol]), "avg": _nums(avg[vcol]), "sem": _nums(avg["sem"]), "n": [int(v) for v in avg["n"]]}


def prepare(raw: pd.DataFrame, um_per_px):
    """Numeric frame/x/y(/z), rows with a missing frame/x/y dropped, coordinates scaled to micrometres."""
    df = pd.DataFrame({c: pd.to_numeric(raw[c], errors="coerce") for c in ("frame", "x", "y")})
    has_z = "z" in raw.columns
    if has_z:
        df["z"] = pd.to_numeric(raw["z"], errors="coerce")
    clean = df.dropna(subset=["frame", "x", "y"]).reset_index(drop=True)
    z_ok = has_z and len(clean) > 0 and bool(np.isfinite(clean["z"].to_numpy(dtype=float)).all())
    if um_per_px is not None:
        for c in ("x", "y", "z") if has_z else ("x", "y"):
            clean[c] = clean[c] * um_per_px
    return clean, len(df) - len(clean), has_z, z_ok


def analyse_condition(name, raw: pd.DataFrame, dt: float, max_intervals: int, sparse_n: int, um_per_px) -> dict:
    """Run every 2D analysis (and Autocorrel_3D if column 7 holds numbers) on one condition."""
    clean, n_dropped, has_z, z_ok = prepare(raw, um_per_px)
    out = {"name": str(name), "n_rows": len(raw), "n_dropped": n_dropped, "has_z": has_z, "z_used": z_ok}
    if len(clean) < 2:
        return {**out, "error": "Fewer than 2 positions with a frame, x and y value."}
    if sparse_n > 1:
        clean = pd.concat([sparse_trajectory(t, sparse_n) for t in split_trajectories(clean)], ignore_index=True)
    dt_eff = dt * sparse_n
    trajs = split_trajectories(clean)
    out["n_tracks"] = len(trajs)
    out["segments"] = [{"x": _nums(t["x"]), "y": _nums(t["y"])} for t in trajs[:MAX_SEGMENTS_RETURNED]]
    out["origin_area"] = [float(v) for v in
                          plot_at_origin.determine_plot_area([plot_at_origin.translate_to_origin(t) for t in trajs])]

    # Speed (run_speed_analysis)
    cell_speeds = [speed.calculate_cell_average_speed(speed.calculate_instantaneous_speeds(t, dt_eff)) for t in trajs]
    cell_speeds = [s for s in cell_speeds if np.isfinite(s)]
    out["speed"] = {"cells": _nums(cell_speeds),
                    "mean": _num(np.mean(cell_speeds)) if cell_speeds else None,
                    "sem": _num(np.std(cell_speeds) / np.sqrt(len(cell_speeds))) if len(cell_speeds) > 1 else None}

    # Dir_Ratio (run_dir_ratio_analysis; last-point mean +- SEM as plot_dir_ratio_last_point shows them)
    dr = [dir_ratio.calculate_directionality_ratio(t, dt_eff) for t in trajs]
    last = [t["dir_ratio"].iloc[-1] for t in dr if len(t) > 1]
    last_s = pd.Series(last, dtype=float)
    out["dir_ratio"] = {**_curve(dir_ratio.align_and_average_dir_ratios(dr), "time", "avg_dir_ratio"),
                        "last": _nums(last), "last_mean": _num(last_s.mean()) if len(last) else None,
                        "last_sem": _num(last_s.sem()) if len(last) else None}

    # MSD (run_msd_analysis)
    msds = [msd.calculate_msd(t, dt_eff, DEFAULTS["msd_max_interval_fraction"]) for t in trajs if len(t) > 1]
    msd_avg = msd.average_msd_results(msds)
    try:
        with np.errstate(all="ignore"):
            alpha, r2 = msd.calculate_alpha_values(msd_avg, DEFAULTS["msd_alpha_fraction"])
    except (np.linalg.LinAlgError, ValueError):
        alpha, r2 = np.nan, np.nan
    n_fit = max(2, int(len(msd_avg) * DEFAULTS["msd_alpha_fraction"])) if len(msd_avg) > 1 else 0
    out["msd"] = {**_curve(msd_avg, "time_interval", "avg_msd"), "alpha": _num(alpha), "r2": _num(r2), "n_fit": n_fit}

    # Autocorrel (run_autocorrelation_analysis)
    vecs = [autocorrel.normalize_vectors(t) for t in trajs]
    ac = [autocorrel.calculate_autocorrelation(v, dt_eff, max_intervals) for v in vecs if len(v) > 1]
    out["autocorr"] = _curve(autocorrel.average_autocorrelation_results(ac), "time_interval", "avg_autocorr") if ac else None

    # Autocorrel_3D (run_autocorrelation_3d_analysis), only with a usable z column
    out["autocorr3d"] = None
    if z_ok:
        v3 = [autocorrel_3d.normalize_vectors_3d(t) for t in trajs]
        ac3 = [autocorrel_3d.calculate_autocorrelation_3d(v, dt_eff, max_intervals) for v in v3 if len(v) > 1]
        if ac3:
            out["autocorr3d"] = _curve(autocorrel_3d.average_autocorrelation_results(ac3), "time_interval", "avg_autocorr")

    # Vel_Cor (run_vel_cor_analysis)
    with np.errstate(all="ignore"):
        vc = [vel_cor.calculate_normalized_velocity_autocorrelation(t, dt_eff, DEFAULTS["vel_cor_max_step_fraction"])
              for t in trajs if len(t) > 2]
    out["vel_cor"] = _curve(vel_cor.average_vel_autocorr_results(vc), "time_interval", "avg_vel_autocorr") if vc else None
    return out


def _safe_name(filename: str) -> str:
    base = os.path.basename(filename or "upload.csv")
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^\w.\- ()]+", "_", stem).strip() or "upload"
    return stem + ext.lower()


def analyse_files(files, time_interval: float, max_intervals: int, sparse_n: int = 1, um_per_px=None) -> dict:
    """files: [(filename, bytes)]. One condition per CSV file, one per Excel sheet. Raises ValueError on bad settings."""
    if not (time_interval > 0):
        raise ValueError("The time interval must be a positive number.")
    if max_intervals < 1 or sparse_n < 1:
        raise ValueError("The number of intervals and the sparse factor must be at least 1.")
    if um_per_px is not None and not (um_per_px > 0):
        raise ValueError("The pixel size must be a positive number (um per pixel).")
    if not files:
        raise ValueError("No file given.")
    conditions, seen = [], {}
    with tempfile.TemporaryDirectory() as d:
        for filename, data in files:
            name = _safe_name(filename)
            if not name.endswith((".csv", ".xlsx", ".xls")):
                raise ValueError(f"{name}: unsupported file type (use .csv, .xlsx or .xls).")
            path = os.path.join(d, name)
            with open(path, "wb") as f:
                f.write(data)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    sheets = load_data(path)
            except Exception as e:  # pandas/openpyxl parse errors, too few columns
                conditions.append({"name": os.path.splitext(name)[0], "error": str(e)})
                continue
            if not sheets:
                conditions.append({"name": os.path.splitext(name)[0], "error": "No sheet has at least 6 columns."})
            for cond, df in sheets.items():
                label = str(cond)
                seen[label] = seen.get(label, 0) + 1
                if seen[label] > 1:
                    label = f"{label} ({seen[label]})"
                conditions.append(analyse_condition(label, df, time_interval, max_intervals, sparse_n, um_per_px))
            os.remove(path)
    return {"engine": "python", "files": [f for f, _ in files], "time_interval": time_interval,
            "max_intervals": max_intervals, "sparse_n": sparse_n, "dt_eff": time_interval * sparse_n,
            "um_per_px": um_per_px, "conditions": conditions}
