"""Write tests/fixtures/diper_fixture.json: small CSV inputs plus the results of the repo's Python analysis on them.

The node test diper/web/static/diper_engine.test.mjs runs the JS port on the same CSV text and asserts agreement;
tests/test_web.py checks the fixture is still what the Python code produces (rtol 1e-9, so it holds on Linux CI too).
Expected values come from diper.web.analysis.analyse_files, i.e. the repo's load_data, split_trajectories, Speed,
Dir_Ratio, MSD, Autocorrel(_3D), Vel_Cor, Plot_At_Origin and Sparse_Data functions.

Inputs are seeded persistent random walks (heading += N(0, sigma) per frame) in the DiPer column layout, plus edge
cases: a one-point track, a stop (zero-length step), a missing value, a non-numeric cell, tracks starting at different
frame numbers, a z column, a stationary cell, too few columns, an empty table and duplicate file names.

Run: python scripts/make_diper_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diper.web.analysis import analyse_files  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "diper_fixture.json"


def walk(rng, n, sigma, step=2.5, z=False):
    th = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, sigma, n))
    x = 100 + np.concatenate([[0], np.cumsum(step * np.cos(th[:-1]))])
    y = 100 + np.concatenate([[0], np.cumsum(step * np.sin(th[:-1]))])
    zz = 10 + np.concatenate([[0], np.cumsum(rng.normal(0, 0.8, n - 1))]) if z else None
    return x, y, zz


def table(rng, sigma, lengths, first_frames=None, z=False, edit=None):
    lines = ["Cell,Experiment,Condition,Frame,X,Y" + (",Z" if z else "")]
    for c, L in enumerate(lengths):
        x, y, zz = walk(rng, L, sigma, z=z)
        f0 = first_frames[c] if first_frames else 1
        for k in range(L):
            row = [f"cell{c + 1}", "exp1", "cond", str(f0 + k), repr(float(x[k])), repr(float(y[k]))]
            if z:
                row.append(repr(float(zz[k])))
            if edit:
                edit(c, k, row)
            lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def persistent_edits(c, k, row):
    if c == 1 and k == 6:      # a stop: same position as the previous frame
        row[4], row[5] = PREV[0], PREV[1]
    if c == 2 and k == 3:      # missing x: row dropped
        row[4] = ""
    if c == 0 and k == 9:      # non-numeric y: row dropped
        row[5] = "n/a?"
    PREV[0], PREV[1] = row[4], row[5]


PREV = ["", ""]


def main():
    rng = np.random.default_rng(20261112)
    stationary = "a,b,c,frame,x,y\n" + "".join(f"s,1,1,{k + 1},50.0,60.0\n" for k in range(8)) \
        + "".join(f"m,1,1,{k + 1},{k * 1.5},{k * 0.5}\n" for k in range(10))
    cases = [
        ("two_conditions", [("persistent.csv", table(rng, 0.3, [30, 25, 18, 1, 12], [1, 5, 1, 1, 40], edit=persistent_edits)),
                            ("random.csv", table(rng, 1.2, [28, 14, 22, 6]))], 0.5, 12, 1, 0.65),
        ("um_3d", [("cells 3D.csv", table(rng, 0.5, [40, 33, 9], z=True))], 2.0, 8, 1, None),
        ("sparse", [("sparse.csv", table(rng, 0.4, [45, 31, 20, 4]))], 1.0, 10, 3, 0.1075),
        ("unweighted", [("tracks.csv", table(rng, 0.7, [50, 12, 7, 26], z=True))], 0.25, 15, 1, 0.5, "tracks"),
        ("edge", [("stationary.csv", stationary), ("few_cols.csv", "frame,x,y\n1,0,0\n2,1,1\n"),
                  ("empty_rows.csv", "a,b,c,frame,x,y\n"), ("stationary.csv", stationary)], 1.0, 5, 1, None),
    ]
    out = []
    for name, files, dt, mi, sn, px, *wt in cases:
        w = wt[0] if wt else "displacements"
        res = analyse_files([(f, t.encode()) for f, t in files], dt, mi, sn, px, w)
        out.append({"name": name, "files": [{"name": f, "text": t} for f, t in files], "time_interval": dt,
                    "max_intervals": mi, "sparse_n": sn, "um_per_px": px, "weighting": w, "expected": res})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_by": "scripts/make_diper_fixture.py", "cases": out}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.1f} kB, {len(out)} cases)")


if __name__ == "__main__":
    main()
