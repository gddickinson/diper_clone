// Methods page: what each analysis computes, as implemented in the repo (diper/*.py), with the paper's equations.
import { route, app, api, esc } from './core.js';

route('methods', async () => {
  const info = await api('/info');
  const d = info.defaults;
  app().innerHTML = `<p class="kicker">Methods</p><h1>What each program computes</h1>
    <p class="lead">The Python package re-implements the DiPer Excel macros. Below, a track has positions r<sub>0</sub> … r<sub>L−1</sub>
      taken Δt apart (the time interval you enter); consecutive rows are treated as consecutive frames.</p>
    <div class="panel"><h3 style="margin-top:0">Tracks and averaging</h3>
      <p>A new track starts wherever the frame number does not increase from one row to the next; one-point tracks are dropped.
        Curves are computed per track, then averaged per time interval over the tracks that reach it: the error bar is
        the sample s.e.m. = standard deviation (ddof 1) / √n, with n the number of tracks (<code>utils.sem</code>), for every program.</p></div>
    <div class="panel"><h3 style="margin-top:0">Speed <span class="tag">speed.py</span></h3>
      <p>Instantaneous speed |r<sub>i</sub> − r<sub>i−1</sub>| / Δt; each cell's average speed is their mean; the grand average is the mean over cells ± s.e.m.</p></div>
    <div class="panel"><h3 style="margin-top:0">Dir_Ratio <span class="tag">dir_ratio.py</span></h3>
      <p class="eq">d/D(t) = |r(t) − r<sub>0</sub>| / Σ |r<sub>i</sub> − r<sub>i−1</sub>|</p>
      <p>Straight-line distance from the start over the path length, plotted against the elapsed time since the track's first
        point (1 at t = 0), averaged across tracks; plus the ratio at each track's last point. The ratio falls faster for
        slower cells, which is why the paper prefers direction autocorrelation for comparing conditions.</p></div>
    <div class="panel"><h3 style="margin-top:0">MSD <span class="tag">msd.py</span></h3>
      <p class="eq">MSD(nΔt) = ⟨ |r<sub>i+n</sub> − r<sub>i</sub>|² ⟩<sub>i</sub></p>
      <p>Overlapping intervals, n = 1 … ⌊${d.msd_max_interval_fraction} L⌋ (half the track, as in the macro). α is the slope of log₁₀ MSD against log₁₀ t over
        the first max(2, ⌊${d.msd_alpha_fraction} × number of averaged points⌋) points, with R²; α ≈ 1 for a random walk, ≈ 2 for straight movement.</p></div>
    <div class="panel"><h3 style="margin-top:0">Autocorrel and Autocorrel_3D <span class="tag">autocorrel.py · autocorrel_3d.py</span></h3>
      <p class="eq">DA(nΔt) = ⟨ û<sub>i</sub> · û<sub>i+n</sub> ⟩<sub>i</sub> = ⟨ cos θ ⟩</p>
      <p>û are unit vectors of the steps r<sub>i+1</sub> − r<sub>i</sub>. Pairs that involve a zero-length step (a stop) are skipped.
        n = 1 … the number of intervals you choose (default ${d.max_intervals}), limited by the track length. Autocorrel_3D does the
        same with x, y, z when the file has a 7th (z) column.</p></div>
    <div class="panel"><h3 style="margin-top:0">Vel_Cor <span class="tag">vel_cor.py</span></h3>
      <p class="eq">v<sub>ac</sub>(n) = ⟨ v<sub>i</sub> · v<sub>i+n</sub> ⟩<sub>i</sub> / Norm, &nbsp; Norm = ⟨ |v<sub>i</sub>|² ⟩</p>
      <p>Velocities instead of unit vectors, normalised by the track's mean squared velocity (paper eqs 6-7), for steps up to
        max(2, ⌊L′/3⌋) with L′ the number of steps. It is robust to stops; with constant speed it equals the direction autocorrelation.</p></div>
    <div class="panel"><h3 style="margin-top:0">Plot_At_Origin and Sparse_Data</h3>
      <p>Plot_At_Origin shifts every track to start at (0, 0) and draws them in a square centred on the origin.
        Sparse_Data keeps 1 frame in N (rows 0, N, 2N … of each track) to remove oversampling noise before the analyses; the
        web pages then use N × Δt as the time interval, as the manual advises.</p></div>
    <div class="panel"><h3 style="margin-top:0">Not in the web pages</h3>
      <p>Autocorrel_NoGaps (replaces stops by tiny random vectors from numpy's random generator) and Make_Charts
        (one image per cell) run in the Python package only; the Analyse page can show any single track instead.</p></div>
    <p class="small">Reference: <a href="${esc(info.reference.url)}" target="_blank" rel="noopener">${esc(info.reference.text)}</a></p>`;
});
