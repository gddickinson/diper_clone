# Session log

## 2026-11-12 — web interface + static GitHub Pages site (branch `web-interface`)

**Added** a FastAPI web app (`python run_web.py`, port 8156) and a static site (`scripts/build_static.py`,
`.github/workflows/pages.yml`), following the direction_autocorrelation_analysis pattern: the user's own track CSVs are
analysed in the browser by a JS port (`diper/web/static/diper_engine.js`) that matches the Python
(`diper/web/analysis.py`, which calls the package's own functions) to ~1e-15 relative on the committed fixture.
Demo data are simulated persistent random walks (labelled SIMULATED everywhere). Units are explicit: pixel size
(µm/px) is required for pixel coordinates, the time unit is chosen, exports record the settings.
Pages: Home, Analyse (+ results: tracks/Plot_At_Origin, Speed, Dir_Ratio, MSD + α, Autocorrel(_3D), Vel_Cor,
summary, 3 CSV exports), Methods, Data format, About. Left out of the web: Autocorrel_NoGaps, Make_Charts files.

**Bugs fixed in the package** (each with a regression test, separate commits):
- `utils.split_trajectories` cut tracks at index labels but sliced by position (wrong tracks for any DataFrame
  whose index is not 0..n-1).
- `vel_cor`: normalised by the SUM of squared velocities instead of the mean (paper eq 6, manual formula), so every
  coefficient was N times too small.
- `dir_ratio`: time was frame × dt, so tracks with different start frames (and Sparse_Data output) were averaged at
  the wrong times; now elapsed time from the track start (row × dt), first point at t = 0.
- `plot_at_origin.determine_plot_area` clipped tracks that all head the same way.
- seaborn used but not declared (ImportError after `pip install -r requirements.txt`); unused import removed from
  dir_ratio.
- `load_data` left column 7 under its own header, so Autocorrel_3D never found 'z' unless the header was exactly 'z'.
- Docs described functions, a `python -m diper.main` CLI and example data that do not exist: corrected.

**Follow-up the same day (George: follow the paper)**, separate "Fix ..." commits:
- Autocorrel, Autocorrel_3D, Autocorrel_NoGaps and Vel_Cor now average cells as in the paper's eq 5, weighted by
  each cell's number of displacements (`weighting="displacements"`, the default); `weighting="tracks"` keeps the
  old plain per-track mean. A shared `utils.average_curves` replaces four copies of the averaging loop. The web pages
  offer both.
- s.e.m. is the sample s.e.m. (ddof=1) everywhere, via `utils.sem`.
- The paper PDF was removed from docs/ (publisher's copyright; still in git history), docs/*.pdf is ignored, and the
  DOI is linked instead.

**Tests** (after the follow-up): `pytest tests` 82 passed; node engine test ok (5 cases, worst 9.77e-16); screenshots.py --static and --base all ok,
no overflow at 390 px.
