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

### Package Definition
- **__init__.py** — Package init with __all__ exports and version info

## Tests: `tests/`
- **test_utils.py** — Tests for trajectory splitting, output dirs, file saving
- **test_autocorrel.py** — Tests for vector normalization, autocorrelation computation
- **test_msd.py** — Tests for MSD computation and averaging

## Data
- **data/test.xlsx** — Example trajectory data for testing

## Configuration
- **pyproject.toml** — Package metadata and build configuration
- **requirements.txt** — Pinned dependency versions
