# DiPer Clone — Roadmap

## Current State
A clean Python reimplementation of the DiPer (Directional Persistence) toolbox for cell migration analysis. Well-organized into ~2,900 lines across 13 files in a proper package structure (`diper/`). Modules cover autocorrelation (2D, 3D, no-gaps), MSD, speed, directionality ratio, velocity correlation, and visualization. Has shared utilities in `utils.py` and example data. No `setup.py` or `pyproject.toml` for installation.

## Short-term Improvements
- [x] Add `setup.py` or `pyproject.toml` so the package is pip-installable
- [x] Fix `init.py` (15 lines) — rename to `__init__.py` if misnamed, and add proper `__all__` exports
- [ ] Add type hints to all public functions in the analysis modules
- [ ] Add input validation in `utils.py` — check for required columns (x, y, frame), NaN handling, minimum track length
- [x] Write unit tests using the example data in `data/` — test each analysis module against known outputs
- [ ] Add docstrings with parameter descriptions and return types to all public functions
- [ ] Add a `__main__.py` entry point for CLI usage: `python -m diper --input data.csv --analysis autocorrel`

## Feature Enhancements
- [ ] Add confidence interval calculation (bootstrap) to `autocorrel.py` and `msd.py`
- [ ] Add persistence time fitting (exponential decay) to direction autocorrelation results
- [ ] Add anomalous diffusion exponent fitting in `msd.py` (MSD ~ t^alpha)
- [ ] Add rose plot / angular histogram visualization for movement directions
- [ ] Add batch processing mode: analyze all files in a directory and produce a summary comparison
- [ ] Add CSV/Excel export with standardized column naming across all analysis types
- [ ] Support tracks with irregular time intervals (interpolation or weighted analysis)

## Long-term Vision
- [ ] Add a lightweight GUI (Tkinter or Streamlit) for interactive trajectory analysis
- [ ] Publish to PyPI as a standalone cell migration analysis package
- [ ] Add integration with TrackMate (ImageJ/Fiji) XML export format
- [ ] Add 3D trajectory visualization with matplotlib 3D or plotly
- [ ] Implement turning angle analysis and velocity autocorrelation as additional metrics
- [ ] Add statistical comparison tests (Mann-Whitney, Kolmogorov-Smirnov) between conditions

## Technical Debt
- [ ] `autocorrel.py` (309 lines), `autocorrel_3d.py` (321 lines), and `autocorrel_nogaps.py` (336 lines) share significant logic — extract common autocorrelation computation into a base function
- [ ] `msd.py` (358 lines) and `dir_ratio.py` (321 lines) likely duplicate trajectory iteration patterns — unify with a shared iterator in `utils.py`
- [ ] `make_charts.py` (161 lines) could use matplotlib style sheets instead of inline formatting
- [ ] `sparse_data.py` (90 lines) is thin — clarify its role or merge into `utils.py`
- [ ] No CI/CD pipeline — add GitHub Actions for linting (ruff) and testing (pytest)
- [x] Missing `requirements.txt` — add with numpy, pandas, matplotlib pinned
