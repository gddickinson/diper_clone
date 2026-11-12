"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diper.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific: the upload endpoint runs the repo's Python analysis ---------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "diper_fixture.json").read_text())


def _post(files, dt=1.0, n=10, sparse=1, px=""):
    return client.post("/api/analyze", files=[("files", (name, text.encode() if isinstance(text, str) else text, "text/csv"))
                                              for name, text in files],
                       data={"time_interval": str(dt), "max_intervals": str(n), "sparse_n": str(sparse), "um_per_px": str(px)})


def _close(a, b, path="", rtol=1e-9):
    """Recursive comparison with a relative tolerance (libm differs in the last digit between macOS and Linux)."""
    if isinstance(b, float) or isinstance(a, float):
        assert a is not None and b is not None and abs(a - b) <= rtol * max(1.0, abs(b)), f"{path}: {a} != {b}"
    elif isinstance(b, dict):
        assert set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in b:
            _close(a[k], b[k], f"{path}.{k}", rtol)
    elif isinstance(b, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]", rtol)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_fixture_is_current(case):
    """The committed fixture is what the Python code gives now (regenerate: python scripts/make_diper_fixture.py)."""
    from diper.web.analysis import analyse_files
    got = analyse_files([(f["name"], f["text"].encode()) for f in case["files"]], case["time_interval"],
                        case["max_intervals"], case["sparse_n"], case["um_per_px"])
    _close(json.loads(json.dumps(got)), case["expected"], case["name"])


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_analyze_endpoint_matches_fixture(case):
    r = _post([(f["name"], f["text"]) for f in case["files"]], case["time_interval"], case["max_intervals"],
              case["sparse_n"], "" if case["um_per_px"] is None else case["um_per_px"])
    assert r.status_code == 200, r.text
    _close(r.json()["conditions"], case["expected"]["conditions"], case["name"])


def test_info_lists_analyses_and_defaults():
    d = client.get("/api/info").json()
    assert d["defaults"]["max_intervals"] == 30 and d["defaults"]["time_interval"] == 1.0
    assert {a["name"] for a in d["analyses"]} >= {"Speed", "Dir_Ratio", "MSD", "Autocorrel", "Vel_Cor"}
    assert d["reference"]["url"] == "https://doi.org/10.1038/nprot.2014.131"


def test_same_numbers_as_run_functions(tmp_path):
    """The bridge gives what the package's own run_* functions give for the same file (CLI-equivalent path)."""
    from diper.autocorrel import run_autocorrelation_analysis
    from diper.msd import run_msd_analysis
    from diper.utils import load_data
    case = FIXTURE["cases"][2]  # 'sparse' file, analysed without sparsing here, in raw units
    path = tmp_path / "cond.csv"
    path.write_text(case["files"][0]["text"])
    data = load_data(str(path))
    ac = run_autocorrelation_analysis(data, 1.0, 10, output_dir=str(tmp_path / "out"))["cond"]
    ms = run_msd_analysis(data, 1.0, output_dir=str(tmp_path / "out"))
    got = _post([("cond.csv", case["files"][0]["text"])], 1.0, 10).json()["conditions"][0]
    np.testing.assert_array_equal(np.array(got["autocorr"]["avg"], float), ac["avg_autocorr"].to_numpy(float))
    np.testing.assert_array_equal(np.array(got["msd"]["avg"], float), ms["msd_results"]["cond"]["avg_msd"].to_numpy(float))
    assert got["msd"]["alpha"] == ms["alpha_values"]["cond"]


def test_pixel_size_scales_lengths_only():
    text = FIXTURE["cases"][2]["files"][0]["text"]
    a = _post([("c.csv", text)]).json()["conditions"][0]
    b = _post([("c.csv", text)], px=0.5).json()["conditions"][0]
    assert b["speed"]["mean"] == pytest.approx(0.5 * a["speed"]["mean"], rel=1e-12)
    assert b["msd"]["avg"][0] == pytest.approx(0.25 * a["msd"]["avg"][0], rel=1e-12)
    assert b["autocorr"]["avg"] == pytest.approx(a["autocorr"]["avg"], rel=1e-12)
    assert b["dir_ratio"]["last"] == pytest.approx(a["dir_ratio"]["last"], rel=1e-12)


def test_excel_sheets_are_conditions(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "t.xlsx"
    with pd.ExcelWriter(p) as w:
        for name, slope in [("straight", 0.0), ("diagonal", 1.0)]:
            pd.DataFrame({"a": 0, "b": 0, "c": 0, "Frame": range(1, 21), "X": np.arange(20.0),
                          "Y": slope * np.arange(20.0)}).to_excel(w, sheet_name=name, index=False)
    r = client.post("/api/analyze", files=[("files", ("t.xlsx", p.read_bytes()))], data={"time_interval": "2", "max_intervals": "5"})
    assert r.status_code == 200, r.text
    conds = r.json()["conditions"]
    assert [c["name"] for c in conds] == ["straight", "diagonal"]
    assert conds[0]["autocorr"]["avg"] == [1, 1, 1, 1, 1] and conds[0]["autocorr"]["t"] == [2, 4, 6, 8, 10]
    assert conds[1]["speed"]["mean"] == pytest.approx(np.sqrt(2) / 2)


def test_bad_inputs():
    good = [("a.csv", "a,b,c,f,x,y\n0,0,0,1,0,0\n0,0,0,2,1,1\n")]
    assert _post([("a.txt", "x")]).status_code == 400
    assert _post(good, dt=0).status_code == 400
    assert _post(good, px=-1).status_code == 400
    assert _post(good, sparse=0).status_code == 400
    c = _post([("few.csv", "f,x,y\n1,2,3\n")]).json()["conditions"][0]
    assert "enough columns" in c["error"]
