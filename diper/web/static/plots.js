// Small SVG plotting helpers (no libraries): line plots with error bars (linear or log-log), trajectory maps with equal
// axes, and per-cell strip plots with mean +- SEM.
import { esc } from './core.js';

export const COLORS = ['#ff7ad9', '#5ee6c8', '#ffc94d', '#8fa8ff', '#ff6b6b', '#b8f05a', '#c792ff', '#5ac8ff'];
const W = 640, H = 380, M = { l: 62, r: 16, t: 16, b: 46 };

export function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
export const fmtTick = v => (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(4)));
let clipN = 0;
const ok = v => v !== null && v !== undefined && Number.isFinite(v);

function axes(xt, yt, sx, sy, xlabel, ylabel, fmtX, fmtY) {
  let g = '';
  for (const v of yt) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(v)}" y2="${sy(v)}" class="grid-l"/><text x="${M.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmtY(v)}</text>`;
  for (const v of xt) g += `<line y1="${M.t}" y2="${H - M.b}" x1="${sx(v)}" x2="${sx(v)}" class="grid-l"/><text x="${sx(v)}" y="${H - M.b + 18}" text-anchor="middle">${fmtX(v)}</text>`;
  g += `<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" class="axis-box"/>`;
  g += `<text x="${(M.l + W - M.r) / 2}" y="${H - 8}" text-anchor="middle" class="axis-label">${esc(xlabel)}</text>`;
  g += `<text transform="translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">${esc(ylabel)}</text>`;
  return g;
}

function legend(items) {
  if (!items.length) return '';
  return `<div class="legend">${items.map(s => `<span><i style="background:${s.color}${s.dash ? ';opacity:.6' : ''}"></i>${esc(s.label)}</span>`).join('')}</div>`;
}

const decades = (lo, hi) => { const o = []; for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) if (e >= lo - 1e-9 && e <= hi + 1e-9) o.push(e); return o; };

// series: [{x, y, err?, color, label?, dash?, width?, opacity?, dots?}]; null y values break the line.
// log: true plots log10(x) and log10(y) (points <= 0 are left out); error bars become y +- err on the log scale.
export function linePlot(el, series, { xlabel = '', ylabel = '', ymin, ymax, xmin, log = false, title = 'plot' } = {}) {
  const tx = v => (log ? (v > 0 ? Math.log10(v) : NaN) : v);
  const pts = series.flatMap(s => s.x.map((x, i) => [tx(x), s.y[i], s.err ? s.err[i] : 0])).filter(p => ok(p[1]) && Number.isFinite(p[0]));
  if (!pts.length) { el.innerHTML = '<p class="mute">Nothing to plot (tracks too short for this analysis).</p>'; return; }
  const lows = pts.map(p => tx(p[1] - (ok(p[2]) ? p[2] : 0))).filter(Number.isFinite), highs = pts.map(p => tx(p[1] + (ok(p[2]) ? p[2] : 0))).filter(Number.isFinite);
  const x0 = xmin ?? (log ? Math.min(...pts.map(p => p[0])) : 0), x1 = Math.max(x0 + 1e-9, ...pts.map(p => p[0]));
  let lo = ymin ?? Math.min(...lows, ...pts.map(p => tx(p[1])).filter(Number.isFinite)), hi = ymax ?? Math.max(...highs);
  if (!(hi > lo)) { hi = lo + 1; }
  const pad = log ? 0.05 * (x1 - x0 || 1) : 0;
  const sx = v => M.l + (v - (x0 - pad)) / ((x1 + pad) - (x0 - pad)) * (W - M.l - M.r);
  const sy = v => H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b);
  const cy = v => Math.max(M.t, Math.min(H - M.b, sy(v)));
  let body = '';
  for (const s of series) {
    let d = '', pen = false;
    s.x.forEach((x, i) => {
      const X = tx(x), Y = ok(s.y[i]) ? tx(s.y[i]) : NaN;
      if (!Number.isFinite(X) || !Number.isFinite(Y)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${sx(X).toFixed(1)},${cy(Y).toFixed(1)}`; pen = true;
    });
    body += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width ?? 2}" stroke-opacity="${s.opacity ?? 1}"${s.dash ? ` stroke-dasharray="${s.dash}"` : ''}/>`;
    if (s.err) s.x.forEach((x, i) => {
      const y = s.y[i], e = s.err[i], X = tx(x);
      if (!ok(y) || !ok(e) || !(e > 0) || !Number.isFinite(X)) return;
      const a = tx(y - e), b = tx(y + e), px = sx(X).toFixed(1);
      const ya = Number.isFinite(a) ? cy(a) : H - M.b, yb = cy(b);
      body += `<path d="M${px},${ya.toFixed(1)}V${yb.toFixed(1)}M${+px - 4},${ya.toFixed(1)}h8M${+px - 4},${yb.toFixed(1)}h8" stroke="${s.color}" stroke-width="1.3"/>`;
    });
    if (s.dots) s.x.forEach((x, i) => { const X = tx(x), Y = ok(s.y[i]) ? tx(s.y[i]) : NaN; if (Number.isFinite(X) && Number.isFinite(Y)) body += `<circle cx="${sx(X).toFixed(1)}" cy="${cy(Y).toFixed(1)}" r="2.8" fill="${s.color}"/>`; });
  }
  const zero = !log && lo < 0 && hi > 0 ? `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(0)}" y2="${sy(0)}" class="zero-l"/>` : '';
  const pow = e => (Math.abs(e) <= 3 ? fmtTick(10 ** e) : `1e${e}`);
  const xt = log ? decades(x0 - pad, x1 + pad) : niceTicks(x0, x1), yt = log ? decades(lo, hi) : niceTicks(lo, hi);
  const g = axes(xt.length ? xt : [x0], yt.length ? yt : [lo], sx, sy, xlabel, ylabel, log ? pow : fmtTick, log ? pow : fmtTick);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="plot" role="img" aria-label="${esc(title)}">${g}${zero}${body}</svg>${legend(series.filter(s => s.label))}`;
}

// Trajectory map with equal axes. segments: [{x, y}] already in plot units. area: [x0, x1, y0, y1] or null (fit).
export function trajectoryPlot(el, segments, { unit = 'µm', title = 'trajectories', color = COLORS[0], area = null, origin = false } = {}) {
  const xs = segments.flatMap(s => s.x), ys = segments.flatMap(s => s.y);
  if (!xs.length) { el.innerHTML = '<p class="mute">No trajectories to draw.</p>'; return; }
  let x0, x1, y0, y1;
  if (area) [x0, x1, y0, y1] = area;
  else {
    const a = Math.min(...xs), b = Math.max(...xs), c = Math.min(...ys), d = Math.max(...ys);
    const span = Math.max(b - a, d - c) || 1, cx = (a + b) / 2, cyy = (c + d) / 2;
    [x0, x1, y0, y1] = [cx - span * 0.55, cx + span * 0.55, cyy - span * 0.55, cyy + span * 0.55];
  }
  const S = 460, m = { l: 58, t: 12, b: 46, r: 12 };
  const sx = v => m.l + (v - x0) / (x1 - x0) * (S - m.l - m.r), sy = v => m.t + (v - y0) / (y1 - y0) * (S - m.t - m.b);
  let g = '';
  for (const v of niceTicks(x0, x1, 5)) g += `<line y1="${m.t}" y2="${S - m.b}" x1="${sx(v)}" x2="${sx(v)}" class="grid-l"/><text x="${sx(v)}" y="${S - m.b + 18}" text-anchor="middle">${fmtTick(v)}</text>`;
  for (const v of niceTicks(y0, y1, 5)) g += `<line x1="${m.l}" x2="${S - m.r}" y1="${sy(v)}" y2="${sy(v)}" class="grid-l"/><text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmtTick(v)}</text>`;
  g += `<rect x="${m.l}" y="${m.t}" width="${S - m.l - m.r}" height="${S - m.t - m.b}" class="axis-box"/>`;
  g += `<text x="${(m.l + S - m.r) / 2}" y="${S - 8}" text-anchor="middle" class="axis-label">${origin ? 'Δx' : 'x'} (${esc(unit)})</text>`;
  g += `<text transform="translate(14 ${(m.t + S - m.b) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">${origin ? 'Δy' : 'y'} (${esc(unit)}, downwards)</text>`;
  const cid = `clip-${++clipN}`;
  g += `<clipPath id="${cid}"><rect x="${m.l}" y="${m.t}" width="${S - m.l - m.r}" height="${S - m.t - m.b}"/></clipPath><g clip-path="url(#${cid})">`;
  segments.forEach((s, i) => {
    const d = s.x.map((x, k) => `${k ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(s.y[k]).toFixed(1)}`).join('');
    const c = Array.isArray(color) ? color[i % color.length] : color;
    g += `<path d="${d}" fill="none" stroke="${c}" stroke-width="1.3" stroke-opacity=".85"/><circle cx="${sx(s.x[0]).toFixed(1)}" cy="${sy(s.y[0]).toFixed(1)}" r="2.2" fill="${c}"/>`;
  });
  g += '</g>';
  if (origin) g += `<circle cx="${sx(0)}" cy="${sy(0)}" r="4" class="origin-dot"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${S} ${S}" class="plot square" role="img" aria-label="${esc(title)}">${g}</svg>`;
}

// Per-cell values by condition: dots (jittered deterministically), mean as a bar, +- SEM whiskers.
// groups: [{label, values, mean, sem, color}]
export function stripPlot(el, groups, { ylabel = '', ymin = 0, ymax, title = 'per-cell values' } = {}) {
  const all = groups.flatMap(g => g.values.filter(ok));
  if (!all.length) { el.innerHTML = '<p class="mute">No values.</p>'; return; }
  const hi = ymax ?? Math.max(...all, ...groups.map(g => (ok(g.mean) ? g.mean + (ok(g.sem) ? g.sem : 0) : 0))) * 1.08;
  const lo = Math.min(ymin, ...all);
  const n = groups.length, bw = (W - M.l - M.r) / n;
  const sx = i => M.l + bw * (i + 0.5), sy = v => H - M.b - (v - lo) / (hi - lo || 1) * (H - M.t - M.b);
  let g = '';
  for (const v of niceTicks(lo, hi)) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(v)}" y2="${sy(v)}" class="grid-l"/><text x="${M.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmtTick(v)}</text>`;
  g += `<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" class="axis-box"/>`;
  g += `<text transform="translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">${esc(ylabel)}</text>`;
  groups.forEach((gr, i) => {
    const X = sx(i), half = Math.min(60, bw * 0.3);
    g += `<text x="${X}" y="${H - M.b + 18}" text-anchor="middle">${esc(gr.label.length > 26 ? gr.label.slice(0, 25) + '…' : gr.label)}</text>`;
    if (ok(gr.mean)) g += `<rect x="${X - half}" y="${sy(Math.max(gr.mean, lo))}" width="${2 * half}" height="${Math.max(0, sy(lo) - sy(gr.mean))}" fill="${gr.color}" fill-opacity=".18" stroke="${gr.color}" stroke-opacity=".6"/>`;
    gr.values.forEach((v, k) => { if (ok(v)) g += `<circle cx="${(X + (((k * 0.618034) % 1) - 0.5) * half * 1.4).toFixed(1)}" cy="${sy(v).toFixed(1)}" r="3" fill="${gr.color}" fill-opacity=".75"/>`; });
    if (ok(gr.mean) && ok(gr.sem)) g += `<path d="M${X},${sy(gr.mean - gr.sem)}V${sy(gr.mean + gr.sem)}M${X - 8},${sy(gr.mean - gr.sem)}h16M${X - 8},${sy(gr.mean + gr.sem)}h16" stroke="var(--ink)" stroke-width="1.6"/>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="plot" role="img" aria-label="${esc(title)}">${g}</svg>`;
}
