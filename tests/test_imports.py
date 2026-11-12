"""
Every analysis module imports with only the declared dependencies (requirements.txt / pyproject.toml).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["autocorrel", "autocorrel_3d", "autocorrel_nogaps", "msd", "dir_ratio", "speed", "vel_cor",
           "make_charts", "plot_at_origin", "sparse_data", "utils"]


def _declared():
    text = (ROOT / "pyproject.toml").read_text()
    return text


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_need_only_declared_packages(name):
    """seaborn is used by speed.py: it must be declared, and modules that do not use it must not import it."""
    declared = _declared()
    code = (
        "import sys, builtins\n"
        f"declared = {declared!r}\n"
        "real = builtins.__import__\n"
        "def guard(n, *a, **k):\n"
        "    top = n.split('.')[0]\n"
        "    if top == 'seaborn' and 'seaborn' not in declared:\n"
        "        raise ImportError('seaborn is not a declared dependency')\n"
        "    return real(n, *a, **k)\n"
        "builtins.__import__ = guard\n"
        f"import diper.{name}\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env={"MPLBACKEND": "Agg", "PATH": ""})
    assert r.returncode == 0, r.stderr[-800:]
