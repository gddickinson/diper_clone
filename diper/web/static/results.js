// Results view: tiles, notes, then one panel per DiPer program, a summary table and CSV exports.
// res has the shape returned by POST /api/analyze (Python) or diper_engine.analyseFiles (JS); lengths are in µm.
import { esc, download } from './core.js';
import { linePlot, trajectoryPlot, stripPlot, COLORS } from './plots.js';
import { summaryCSV, curvesCSV, cellsCSV } from './export.js';
import { DEMO, theoryCurve } from './diper_sim.js';

const f3 = v => (v === null || v === undefined ? '—' : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(3));
const pm = (m, s) => (m === null || m === undefined ? '—' : `${f3(m)}${s === null || s === undefined ? '' : ` ± ${f3(s)}`}`);
const col = i => COLORS[i % COLORS.length];

function notes(res, meta, ok, bad) {
  const out = [];
  if (meta.simulated) out.push('<span class="sim-badge">SIMULATED</span> Results from simulated persistent random walks, not measurements. Dashed lines on the autocorrelation plots show the expected curve exp(−m σ²/2) for a lag of m frames.');
  bad.forEach(c => out.push(`<span class="err">${esc(c.name)}: ${esc(c.error)}</span>`));
  ok.filter(c => c.n_dropped > 0).forEach(c => out.push(`${esc(c.name)}: ${c.n_dropped} row(s) with a missing or non-numeric frame, x or y were dropped.`));
  ok.filter(c => c.has_z && !c.z_used).forEach(c => out.push(`${esc(c.name)}: the 7th (z) column has missing or non-numeric values, so Autocorrel_3D was skipped.`));
  ok.filter(c => c.n_tracks > meta.maxSegments).forEach(c => out.push(`${esc(c.name)}: the track plots show the first ${meta.maxSegments} of ${c.n_tracks} tracks; the analyses use all of them.`));
  if (res.sparse_n > 1) out.push(`Sparse_Data: 1 of every ${res.sparse_n} frames kept, so the time interval used is ${res.sparse_n} × ${res.time_interval} = ${+res.dt_eff.toPrecision(10)} ${esc(meta.timeUnit)}.`);
  out.push(res.weighting === 'tracks'
    ? 'Autocorrel and Vel_Cor: plain mean of the per-cell curves (option chosen; the paper weights cells by their number of displacements).'
    : 'Autocorrel and Vel_Cor: cells weighted by their number of displacements, as in the paper (eq 5).');
  out.push(res.um_per_px ? `Coordinates in pixels × ${res.um_per_px} µm/px, so all lengths below are in µm.` : 'Coordinates taken as µm.');
  if (meta.check !== null && meta.check !== undefined) {
    out.push(Number.isFinite(meta.check) && meta.check < 1e-9
      ? `<span class="ok">Cross-check:</span> the in-browser JS port gives the same numbers as Python here (largest relative difference ${meta.check.toExponential(1)}).`
      : `<span class="err">Cross-check:</span> the JS port differs from Python on these files (${meta.check}); the Python numbers are shown.`);
  }
  return out.map(n => `<p class="small">${n}</p>`).join('');
}

function panel(id, title, tag, body) {
  return `<div class="panel" id="${id}"><h3 style="margin-top:0">${title} <span class="tag">${tag}</span></h3>${body}</div>`;
}

export function renderResults(host, res, meta) {
  const ok = res.conditions.filter(c => !c.error), bad = res.conditions.filter(c => c.error);
  const tu = meta.timeUnit, nTr = ok.reduce((s, c) => s + c.n_tracks, 0), nPts = ok.reduce((s, c) => s + c.n_rows - c.n_dropped, 0);
  const any3d = ok.some(c => c.autocorr3d);
  host.innerHTML = `<h2>Results</h2>
    <div class="tiles"><div class="tile"><b>${ok.length}</b><span>condition${ok.length === 1 ? '' : 's'}</span></div>
      <div class="tile"><b>${nTr}</b><span>tracks</span></div><div class="tile"><b>${nPts}</b><span>positions read</span></div>
      <div class="tile"><b>${res.engine === 'python' ? 'Python' : 'JS'}</b><span>engine</span></div></div>
    ${notes(res, meta, ok, bad)}
    ${ok.length ? `
    ${panel('p-tracks', 'Trajectories', 'Plot_At_Origin', `<div class="row">
        <div><label for="v-cond">Condition</label><select id="v-cond">${ok.map((c, i) => `<option value="${i}">${esc(c.name)}</option>`).join('')}</select></div>
        <div><label for="v-mode">Show</label><select id="v-mode"><option value="origin">all tracks from the origin</option><option value="raw">all tracks as recorded</option><option value="one">one track</option></select></div>
        <div><label for="v-track">Track</label><select id="v-track"></select></div></div>
      <div id="g-tracks"></div><p id="tracks-note" class="small mute"></p>`)}
    <div class="two">
      ${panel('p-speed', 'Speed', 'speed.py', `<div id="g-speed"></div><p class="small mute">Average speed of each cell (dots), mean (bar) ± s.e.m. (µm/${esc(tu)}).</p>`)}
      ${panel('p-last', 'Dir_Ratio at the last point', 'dir_ratio.py', `<div id="g-last"></div><p class="small mute">d/D at each track's last point, mean ± s.e.m. Depends on track duration: compare the curve below.</p>`)}
    </div>
    <div class="two">
      ${panel('p-dr', 'Dir_Ratio over time', 'dir_ratio.py', `<div id="g-dr"></div><p class="small mute">Straight-line distance / path length versus elapsed time; mean ± s.e.m. over tracks.</p>`)}
      ${panel('p-msd', 'MSD', 'msd.py', `<label><input id="v-msdlog" type="checkbox" checked> log-log axes</label><div id="g-msd"></div><p class="small mute">Mean square displacement ± s.e.m. (µm²); α in the legend.</p>`)}
    </div>
    <div class="two">
      ${panel('p-ac', 'Direction autocorrelation', any3d ? 'autocorrel.py · autocorrel_3d.py' : 'autocorrel.py', `${any3d ? '<label><input id="v-3d" type="checkbox"> 3D (x, y, z)</label>' : ''}<div id="g-ac"></div><p class="small mute">Mean cos θ between step directions a time interval apart (population average, see notes) ± s.e.m. over tracks.</p>`)}
      ${panel('p-vc', 'Velocity autocorrelation', 'vel_cor.py', `<div id="g-vc"></div><p class="small mute">⟨v(t)·v(t+τ)⟩ / ⟨v²⟩ per track, mean ± s.e.m. over tracks.</p>`)}
    </div>
    ${panel('p-sum', 'Summary', 'all programs', `<div class="scroll" id="t-sum"></div>
      <button id="x-sum" type="button">Export summary CSV</button><button id="x-cur" class="alt" type="button">Export all curves CSV</button><button id="x-cell" class="alt" type="button">Export per-cell CSV</button>`)}` : ''}`;
  if (!ok.length) return;
  const $ = s => host.querySelector(s);

  // Trajectories
  const fillTracks = () => {
    const c = ok[+$('#v-cond').value];
    $('#v-track').innerHTML = c.segments.map((_, k) => `<option value="${k}">track ${k + 1}</option>`).join('');
  };
  const drawTracks = () => {
    const i = +$('#v-cond').value, c = ok[i], mode = $('#v-mode').value;
    $('#v-track').disabled = mode !== 'one';
    let segs = c.segments, opt = { color: col(i), title: `Trajectories of ${c.name}` };
    if (mode === 'one') segs = [c.segments[+$('#v-track').value || 0]];
    if (mode === 'origin') {
      segs = segs.map(s => ({ x: s.x.map(v => v - s.x[0]), y: s.y.map(v => v - s.y[0]) }));
      opt = { ...opt, area: c.origin_area, origin: true, color: COLORS };
    }
    trajectoryPlot($('#g-tracks'), segs, opt);
    $('#tracks-note').textContent = mode === 'origin'
      ? `${segs.length} tracks shifted to start at (0, 0); square area ±${c.origin_area[1]} µm as determine_plot_area gives it.`
      : mode === 'raw' ? `${segs.length} tracks; dots mark track starts.` : 'One track (what Make_Charts draws per cell); the dot marks its start.';
  };
  $('#v-cond').onchange = () => { fillTracks(); drawTracks(); };
  $('#v-mode').onchange = drawTracks; $('#v-track').onchange = drawTracks;
  fillTracks(); drawTracks();

  // Per-cell strip plots
  stripPlot($('#g-speed'), ok.map((c, i) => ({ label: c.name, values: c.speed.cells, mean: c.speed.mean, sem: c.speed.sem, color: col(i) })),
    { ylabel: `average speed (µm/${tu})`, title: 'Average speed per cell' });
  stripPlot($('#g-last'), ok.map((c, i) => ({ label: c.name, values: c.dir_ratio.last, mean: c.dir_ratio.last_mean, sem: c.dir_ratio.last_sem, color: col(i) })),
    { ylabel: 'd/D at last point', ymax: 1.05, title: 'Directionality ratio at the last point' });

  const curve = (c, key, i, extra = {}) => (c[key] ? { x: c[key].t, y: c[key].avg, err: c[key].sem, color: col(i), label: c.name, width: 2.2, dots: true, ...extra } : null);
  linePlot($('#g-dr'), ok.map((c, i) => curve(c, 'dir_ratio', i)).filter(Boolean), { xlabel: `elapsed time (${tu})`, ylabel: 'directionality ratio d/D', ymin: 0, ymax: 1.05, title: 'Directionality ratio over time' });
  const drawMsd = () => {
    const log = $('#v-msdlog').checked;
    linePlot($('#g-msd'), ok.map((c, i) => curve(c, 'msd', i, { label: `${c.name} (α ${c.msd.alpha === null ? '—' : c.msd.alpha.toFixed(2)})` })).filter(Boolean),
      { xlabel: `time interval (${tu})`, ylabel: 'MSD (µm²)', log, ymin: log ? undefined : 0, title: 'Mean square displacement' });
  };
  $('#v-msdlog').onchange = drawMsd; drawMsd();
  const theory = (c, key, i) => {
    const sim = meta.simulated && DEMO.conditions.find(dc => c.name === dc.name);
    return sim && c[key] ? { x: c[key].t, y: theoryCurve(sim.sigma, c[key].t.map(t => t / res.dt_eff * res.sparse_n)), color: col(i), dash: '6 5', width: 1.4, label: `expected, ${c.name}` } : null;
  };
  const drawAc = () => {
    const key = $('#v-3d') && $('#v-3d').checked ? 'autocorr3d' : 'autocorr';
    linePlot($('#g-ac'), ok.flatMap((c, i) => [curve(c, key, i), key === 'autocorr' ? theory(c, key, i) : null]).filter(Boolean),
      { xlabel: `time interval (${tu})`, ylabel: 'direction autocorrelation', ymin: -0.2, ymax: 1.05, title: 'Direction autocorrelation' });
  };
  if ($('#v-3d')) $('#v-3d').onchange = drawAc;
  drawAc();
  linePlot($('#g-vc'), ok.flatMap((c, i) => [curve(c, 'vel_cor', i), theory(c, 'vel_cor', i)]).filter(Boolean),
    { xlabel: `time interval (${tu})`, ylabel: 'normalized velocity autocorrelation', ymin: -0.2, ymax: 1.05, title: 'Velocity autocorrelation' });

  // Summary table
  const first = (c, key) => (c[key] && c[key].avg.length ? `${pm(c[key].avg[0], c[key].sem[0])} <span class="mute">at ${+c[key].t[0].toPrecision(6)} ${esc(tu)}</span>` : '—');
  $('#t-sum').innerHTML = `<table><tr><th>condition</th><th>tracks</th><th>speed (µm/${esc(tu)})</th><th>d/D last point</th><th>MSD α (R²)</th><th>autocorrelation, first interval</th><th>Vel_Cor, first interval</th></tr>
    ${ok.map(c => `<tr><td>${esc(c.name)}</td><td class="mono">${c.n_tracks}</td><td class="mono">${pm(c.speed.mean, c.speed.sem)}</td>
      <td class="mono">${pm(c.dir_ratio.last_mean, c.dir_ratio.last_sem)}</td>
      <td class="mono">${c.msd.alpha === null ? '—' : `${c.msd.alpha.toFixed(3)} (${c.msd.r2 === null ? '—' : c.msd.r2.toFixed(3)}, ${c.msd.n_fit} pts)`}</td>
      <td class="mono">${first(c, 'autocorr')}</td><td class="mono">${first(c, 'vel_cor')}</td></tr>`).join('')}</table>`;
  const base = meta.simulated ? 'diper_simulated_demo' : 'diper';
  $('#x-sum').onclick = () => download(`${base}_summary.csv`, summaryCSV(res, meta), 'text/csv');
  $('#x-cur').onclick = () => download(`${base}_curves.csv`, curvesCSV(res, meta), 'text/csv');
  $('#x-cell').onclick = () => download(`${base}_per_cell.csv`, cellsCSV(res, meta), 'text/csv');
}
