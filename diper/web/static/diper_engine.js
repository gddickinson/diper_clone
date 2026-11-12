// DiPer engine: a pure-JS port of the repo's Python analyses, used on the static site (and as a cross-check locally).
// Mirrors diper/web/analysis.py step by step, which calls the repo's own functions:
//   utils.split_trajectories   a new track starts wherever the frame number does not increase; 1-point tracks dropped
//   sparse_data                keep rows 0, N, 2N... of each track; time interval becomes N x dt
//   plot_at_origin             square area centred on (0, 0), half-width = ceil(1.1 x max |offset| / 10) x 10
//   speed                      mean of step length / dt per cell; grand mean +- s.e.m.
//   dir_ratio                  d/D over elapsed time (row x dt), first point 1; averaged by time; last-point mean +- SEM
//   msd                        overlapping intervals up to floor(L / 2) steps; alpha = slope of log10 MSD vs log10 t
//                              over the first max(2, floor(0.1 x n)) averaged points
//   autocorrel(_3d)            mean dot product of unit step vectors `step` apart (zero-length steps skipped)
//   vel_cor                    <v_i . v_i+n> / <|v|^2> per track, steps up to max(2, floor(N / 3))
// Autocorrel(_3D) and Vel_Cor are averaged over cells as in the paper's eq 5 (weights = displacements per cell), or
// as the plain mean of per-cell curves with weighting 'tracks' (utils.average_curves / track_weights).
// Error bars are the sample s.e.m. (utils.sem: std ddof 1 / sqrt(n)), n = number of tracks.
import { npSum, npMean, sem, pdMean, pdSem, lineFit } from './diper_num.js';
import { readCondition } from './diper_csv.js';

export const MAX_SEGMENTS_RETURNED = 300;
const MSD_FRACTION = 0.5, ALPHA_FRACTION = 0.1, VEL_COR_FRACTION = 1 / 3;
const num = v => (Number.isFinite(v) ? v : null);
const nums = a => a.map(num);

// --- tracks -----------------------------------------------------------------------------------------------------
function slice(t, a, b) {
  const o = { frame: t.frame.slice(a, b), x: t.x.slice(a, b), y: t.y.slice(a, b) };
  if (t.z) o.z = t.z.slice(a, b);
  return o;
}

export function splitTrajectories(t) {
  const n = t.frame.length;
  if (n <= 1) return n ? [t] : [];
  const starts = [0];
  for (let i = 1; i < n; i++) if (t.frame[i] - t.frame[i - 1] <= 0) starts.push(i);
  starts.push(n);
  const out = [];
  for (let k = 0; k < starts.length - 1; k++) if (starts[k + 1] - starts[k] > 1) out.push(slice(t, starts[k], starts[k + 1]));
  return out.length ? out : [t];
}

function concat(tracks, hasZ) {
  const o = { frame: [], x: [], y: [], z: hasZ ? [] : null };
  for (const t of tracks) { o.frame.push(...t.frame); o.x.push(...t.x); o.y.push(...t.y); if (hasZ) o.z.push(...t.z); }
  return o;
}

export function sparseTrajectory(t, n) {
  if (t.frame.length <= 1 || n <= 1) return t;
  const keep = (a) => a.filter((_, i) => i % n === 0);
  const o = { frame: keep(t.frame), x: keep(t.x), y: keep(t.y) };
  if (t.z) o.z = keep(t.z);
  return o;
}

// --- averaging over tracks (average_*_results / align_and_average_dir_ratios) ------------------------------------
export function averageCurves(curves, weights = null) {
  const times = [...new Set(curves.flatMap(c => c.t))].sort((a, b) => a - b);
  const maps = curves.map(c => new Map(c.t.map((t, i) => [t, i])));
  const avg = [], sems = [], n = [];
  for (const t of times) {
    const vals = [], w = [];
    curves.forEach((c, j) => { const i = maps[j].get(t); if (i !== undefined) { vals.push(c.v[i]); if (weights) w.push(weights[j]); } });
    // np.average: sum(v * w) / sum(w)
    avg.push(!vals.length ? NaN : weights ? npSum(vals.map((v, k) => v * w[k])) / npSum(w) : npMean(vals));
    sems.push(sem(vals));
    n.push(vals.length);
  }
  return { t: times, avg, sem: sems, n };
}
const jsonCurve = c => ({ t: nums(c.t), avg: nums(c.avg), sem: nums(c.sem), n: c.n });

// --- analyses on one track ----------------------------------------------------------------------------------------
export function cellAverageSpeed(t, dt) {
  const v = [];
  for (let i = 1; i < t.x.length; i++) {
    const dx = t.x[i] - t.x[i - 1], dy = t.y[i] - t.y[i - 1], s = Math.sqrt(dx * dx + dy * dy) / dt;
    if (Number.isFinite(s)) v.push(s);
  }
  return t.x.length <= 1 || !v.length ? NaN : pdMean(v);
}

export function dirRatio(t, dt) {
  const n = t.x.length, ratio = [1], time = t.x.map((_, i) => i * dt);
  let path = 0;
  for (let i = 1; i < n; i++) {
    const dx = t.x[i] - t.x[i - 1], dy = t.y[i] - t.y[i - 1];
    path += Math.sqrt(dx * dx + dy * dy);
    const ex = t.x[i] - t.x[0], ey = t.y[i] - t.y[0];
    ratio.push(Math.sqrt(ex * ex + ey * ey) / path);
  }
  return { t: time, v: ratio };
}

export function msdCurve(t, dt, fraction = MSD_FRACTION) {
  const L = t.x.length, maxStep = Math.trunc(L * fraction), out = { t: [], v: [] };
  if (L <= 1) return out;
  for (let step = 1; step <= maxStep; step++) {
    if (L - step <= 0) continue;
    const sq = [];
    for (let i = 0; i < L - step; i++) {
      const dx = t.x[i + step] - t.x[i], dy = t.y[i + step] - t.y[i];
      sq.push(dx * dx + dy * dy);
    }
    out.t.push(step * dt); out.v.push(npMean(sq));
  }
  return out;
}

export function alphaFit(avg, fraction = ALPHA_FRACTION) {
  if (avg.t.length <= 1) return { alpha: NaN, r2: NaN, n_fit: 0 };
  const nFit = Math.max(2, Math.trunc(avg.t.length * fraction));
  const lx = avg.t.slice(0, nFit).map(Math.log10), ly = avg.avg.slice(0, nFit).map(Math.log10);
  if (![...lx, ...ly].every(Number.isFinite)) return { alpha: NaN, r2: NaN, n_fit: nFit };
  const f = lineFit(lx, ly);
  return { alpha: f.slope, r2: f.r2, n_fit: nFit };
}

// Unit step vectors (normalize_vectors / normalize_vectors_3d): zero-length steps give the zero vector.
export function unitVectors(t, use3d = false) {
  const v = [];
  for (let i = 0; i < t.x.length - 1; i++) {
    const dx = t.x[i + 1] - t.x[i], dy = t.y[i + 1] - t.y[i], dz = use3d ? t.z[i + 1] - t.z[i] : 0;
    const m = use3d ? Math.sqrt(dx * dx + dy * dy + dz * dz) : Math.sqrt(dx * dx + dy * dy);
    v.push(m > 0 ? [dx / m, dy / m, dz / m] : [0, 0, 0]);
  }
  return v;
}

export function autocorrCurve(vecs, dt, maxIntervals, use3d = false) {
  const out = { t: [], v: [] }, L = vecs.length;
  if (L <= 1) return out;
  const zero = u => u[0] === 0 && u[1] === 0 && (!use3d || u[2] === 0);
  const maxI = Math.min(maxIntervals, L - 1);
  for (let step = 1; step <= maxI; step++) {
    const c = [];
    for (let i = 0; i < L - step; i++) {
      const a = vecs[i], b = vecs[i + step];
      if (zero(a) || zero(b)) continue;
      c.push(use3d ? a[0] * b[0] + a[1] * b[1] + a[2] * b[2] : a[0] * b[0] + a[1] * b[1]);
    }
    if (c.length) { out.t.push(step * dt); out.v.push(npMean(c)); }
  }
  return out;
}

export function velCorCurve(t, dt, fraction = VEL_COR_FRACTION) {
  const out = { t: [], v: [] };
  if (t.x.length <= 2) return out;
  const dx = [], dy = [], sq = [];
  for (let i = 1; i < t.x.length; i++) { dx.push(t.x[i] - t.x[i - 1]); dy.push(t.y[i] - t.y[i - 1]); }
  for (let i = 0; i < dx.length; i++) sq.push(dx[i] * dx[i] + dy[i] * dy[i]);
  const norm = npSum(sq) / (dx.length * (dt * dt));  // eq 6: mean squared velocity
  const maxStep = Math.max(2, Math.trunc(dx.length * fraction));
  for (let step = 1; step <= maxStep; step++) {
    let sum = 0, count = 0;
    for (let i = 0; i < dx.length - step; i++) {
      sum += (dx[i] * dx[i + step] + dy[i] * dy[i + step]) / (dt * dt * norm);
      count++;
    }
    if (count) { out.t.push(step * dt); out.v.push(sum / count); }
  }
  return out;
}

export function originArea(tracks) {
  if (!tracks.length) return [-10, 10, -10, 10];
  let ext = 0;
  for (const t of tracks) {
    const rx = t.x.map(v => v - t.x[0]), ry = t.y.map(v => v - t.y[0]);
    ext = Math.max(ext, Math.abs(Math.min(...rx)), Math.abs(Math.max(...rx)), Math.abs(Math.min(...ry)), Math.abs(Math.max(...ry)));
  }
  const half = Math.ceil(1.1 * ext / 10) * 10;
  return half === 0 ? [-10, 10, -10, 10] : [-half, half, -half, half];
}

// --- one condition (analysis.analyse_condition) ------------------------------------------------------------------
export const WEIGHTINGS = ['displacements', 'tracks'];
const trackWeights = (weighting, nDisp) => (weighting === 'displacements' ? nDisp : null);

export function analyseCondition(cond, dt, maxIntervals, sparseN, umPerPx, weighting = 'displacements') {
  const hasZ = !!cond.z, keep = [];
  for (let i = 0; i < cond.frame.length; i++) if (!Number.isNaN(cond.frame[i]) && !Number.isNaN(cond.x[i]) && !Number.isNaN(cond.y[i])) keep.push(i);
  const pick = a => keep.map(i => a[i]);
  let clean = { frame: pick(cond.frame), x: pick(cond.x), y: pick(cond.y), z: hasZ ? pick(cond.z) : null };
  const zOk = hasZ && keep.length > 0 && clean.z.every(Number.isFinite);
  if (umPerPx !== null && umPerPx !== undefined) {
    for (const c of hasZ ? ['x', 'y', 'z'] : ['x', 'y']) clean[c] = clean[c].map(v => v * umPerPx);
  }
  const out = { name: cond.name, n_rows: cond.n_rows, n_dropped: cond.n_rows - keep.length, has_z: hasZ, z_used: zOk };
  if (keep.length < 2) return { ...out, error: 'Fewer than 2 positions with a frame, x and y value.' };
  if (sparseN > 1) clean = concat(splitTrajectories(clean).map(t => sparseTrajectory(t, sparseN)), hasZ);
  const dtE = dt * sparseN, trajs = splitTrajectories(clean);
  out.n_tracks = trajs.length;
  out.segments = trajs.slice(0, MAX_SEGMENTS_RETURNED).map(t => ({ x: nums(t.x), y: nums(t.y) }));
  out.origin_area = originArea(trajs);

  const speeds = trajs.map(t => cellAverageSpeed(t, dtE)).filter(Number.isFinite);
  out.speed = { cells: nums(speeds), mean: speeds.length ? num(npMean(speeds)) : null,
    sem: speeds.length > 1 ? num(sem(speeds)) : null };

  const dr = trajs.map(t => dirRatio(t, dtE)), last = dr.filter(d => d.v.length > 1).map(d => d.v[d.v.length - 1]);
  out.dir_ratio = { ...jsonCurve(averageCurves(dr)), last: nums(last),
    last_mean: last.length ? num(pdMean(last)) : null, last_sem: last.length ? num(pdSem(last)) : null };

  const msdAvg = averageCurves(trajs.filter(t => t.x.length > 1).map(t => msdCurve(t, dtE)));
  const fit = alphaFit(msdAvg);
  out.msd = { ...jsonCurve(msdAvg), alpha: num(fit.alpha), r2: num(fit.r2), n_fit: fit.n_fit };

  const vecs = trajs.map(t => unitVectors(t)).filter(v => v.length > 1), ac = vecs.map(v => autocorrCurve(v, dtE, maxIntervals));
  out.autocorr = ac.length ? jsonCurve(averageCurves(ac, trackWeights(weighting, vecs.map(v => v.length)))) : null;
  out.autocorr3d = null;
  if (zOk) {
    const v3 = trajs.map(t => unitVectors(t, true)).filter(v => v.length > 1), ac3 = v3.map(v => autocorrCurve(v, dtE, maxIntervals, true));
    if (ac3.length) out.autocorr3d = jsonCurve(averageCurves(ac3, trackWeights(weighting, v3.map(v => v.length))));
  }
  const used = trajs.filter(t => t.x.length > 2), vc = used.map(t => velCorCurve(t, dtE));
  out.vel_cor = vc.length ? jsonCurve(averageCurves(vc, trackWeights(weighting, used.map(t => t.x.length - 1)))) : null;
  return out;
}

// --- files (analysis.analyse_files): [{name, text}] -> the JSON shape of POST /api/analyze ----------------------
export function analyseFiles(files, timeInterval, maxIntervals, sparseN = 1, umPerPx = null, weighting = 'displacements') {
  const dt = +timeInterval, mi = +maxIntervals, sn = +sparseN, px = umPerPx === null || umPerPx === '' ? null : +umPerPx;
  if (!(dt > 0)) throw new Error('The time interval must be a positive number.');
  if (!(mi >= 1) || !(sn >= 1)) throw new Error('The number of intervals and the sparse factor must be at least 1.');
  if (px !== null && !(px > 0)) throw new Error('The pixel size must be a positive number (um per pixel).');
  if (!WEIGHTINGS.includes(weighting)) throw new Error(`weighting must be one of ${WEIGHTINGS.join(', ')}.`);
  if (!files.length) throw new Error('No file given.');
  const conditions = [], seen = new Map();
  for (const f of files) {
    if (!/\.csv$/i.test(f.name)) throw new Error(`${f.name}: Excel files need the local app (Python); save each sheet as CSV.`);
    let cond;
    try { cond = readCondition(f.text, f.name); } catch (e) { conditions.push({ name: f.name.replace(/\.[^.]+$/, ''), error: e.message }); continue; }
    const k = (seen.get(cond.name) || 0) + 1;
    seen.set(cond.name, k);
    if (k > 1) cond.name = `${cond.name} (${k})`;
    conditions.push(analyseCondition(cond, dt, mi, sn, px, weighting));
  }
  return { engine: 'js', files: files.map(f => f.name), time_interval: dt, max_intervals: mi, sparse_n: sn,
    dt_eff: dt * sn, um_per_px: px, weighting, conditions };
}

// Largest relative difference between two results of the same shape (Infinity if the shapes or null-ness differ).
export function maxDifference(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) / Math.max(1, Math.abs(b));
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return a === b ? 0 : Infinity;
  if (Array.isArray(a) !== Array.isArray(b)) return Infinity;
  const keys = Array.isArray(a) ? (a.length === b.length ? a.map((_, i) => i) : null) : Object.keys(b).filter(k => k !== 'engine' && k !== 'files');
  if (!keys) return Infinity;
  let d = 0;
  for (const k of keys) d = Math.max(d, maxDifference(a[k], b[k]));
  return d;
}
