# DiPer Clone — Interface Map

## Package Structure: `diper/`

### Core Analysis Modules
- **autocorrel.py** — 2D direction autocorrelation analysis
  - `normalize_vectors(traj)` -> DataFrame with x_vec, y_vec
  - `calculate_autocorrelation(traj, time_interval, max_intervals)` -> DataFrame
  - `average_autocorrelation_results(all_autocorrs)` -> DataFrame
  - `run_autocorrelation_analysis(data, time_interval, ...)` -> dict
  - `plot_autocorrelation(results, output_dir)` -> Figure

- **autocorrel_3d.py** — 3D direction autocorrelation (extends autocorrel to 3D)

- **autocorrel_nogaps.py** — Direction autocorrelation without gap handling

- **msd.py** — Mean Square Displacement analysis
  - `calculate_msd(traj, time_interval, max_interval_fraction)` -> DataFrame
  - `calculate_alpha_values(msd_results, fraction)` -> (alpha, r_squared)
  - `average_msd_results(all_msds)` -> DataFrame
  - `run_msd_analysis(data, time_interval, ...)` -> dict

- **dir_ratio.py** — Directionality ratio analysis

- **speed.py** — Speed analysis

- **vel_cor.py** — Velocity correlation analysis

### Utilities and Visualization
- **utils.py** — Shared utilities
  - `load_data(file_path)` -> dict of DataFrames
  - `split_trajectories(df)` -> list of DataFrames
  - `ensure_output_dir(output_dir, subdir)` -> str
  - `save_figure(fig, output_dir, filename, formats)`
  - `save_results(df, output_dir, filename)`

- **make_charts.py** — Chart generation utilities

- **plot_at_origin.py** — Plot trajectories aligned at origin

- **sparse_data.py** — Sparse/gapped data handling

### Web interface: `diper/web/` (python run_web.py -> http://127.0.0.1:8156)
- **web/app.py** — FastAPI app: `/api/info` (defaults, analyses, reference), `POST /api/analyze` (uploads);
  `STATIC_ENDPOINTS` for the static export
- **web/analysis.py** — bridge to the package: `analyse_files(files, time_interval, max_intervals, sparse_n, um_per_px)`
  -> `analyse_condition` (load_data, drop rows, scale px->um, Sparse_Data, split_trajectories, then the
  per-track functions of speed/dir_ratio/msd/autocorrel/autocorrel_3d/vel_cor/plot_at_origin)
- **web/static/** — ES-module front end, no build step:
  - `core.js`, `style.css` (shared web kit), `style_app.css`, `index.html`, `main.js` (router), `static_api.js`
    (static site: answers `api()` from exported JSON and runs the JS engine)
  - pages: `home.js`, `analyse.js` (inputs) -> `results.js` (panels) + `export.js` (CSV), `methods.js`,
    `format.js`, `about.js`; `plots.js` (SVG line/log-log, trajectory and strip plots)
  - engine: `diper_engine.js` (JS port of analysis.py), `diper_num.js` (numpy/pandas summation order),
    `diper_csv.js` (load_data CSV branch), `diper_sim.js` (SIMULATED demo tracks), `diper_engine.test.mjs` (node)

### Package Definition
- **__init__.py** — Package init with __all__ exports and version info

## Scripts
- **run_web.py** — starts the web app (uvicorn, port 8156)
- **scripts/build_static.py** — exports the static GitHub Pages site to site/
- **scripts/screenshots.py** — Playwright smoke test of every route (+ README images in docs/img/)
- **scripts/make_diper_fixture.py** — writes tests/fixtures/diper_fixture.json (Python results for the JS test)

## Tests: `tests/`
- **test_utils.py** — Tests for trajectory splitting, output dirs, file saving
- **test_autocorrel.py** — Tests for vector normalization, autocorrelation computation
- **test_msd.py** — Tests for MSD computation and averaging
- **test_vel_cor.py**, **test_dir_ratio.py**, **test_plot_at_origin.py** — regression tests for the 2026-11-12 fixes
- **test_imports.py** — every module imports with only the declared dependencies
- **test_web.py** — web API, static build, fixture currency (rtol 1e-9), Python == run_* functions, node engine test

## Data
- `data/` is git-ignored: no example data is committed (the web interface uses simulated demo tracks)

## Configuration
- **pyproject.toml** — Package metadata and build configuration
- **requirements.txt** — Dependency ranges; **requirements-web.txt** — pinned web/test deps
- **.github/workflows/pages.yml** — tests + static site to GitHub Pages on push to main
