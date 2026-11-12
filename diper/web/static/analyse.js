// Analyse page: load CSVs (or the simulated demo), set time interval / units / intervals / sparsing, run the analyses.
// Local app: POST api/analyze runs the repo's Python (diper/web/analysis.py). Static site: static_api.js runs diper_engine.js.
import { route, app, api, esc, isStatic, download } from './core.js';
import { simulateFiles, DEMO } from './diper_sim.js';
import { analyseFiles, maxDifference } from './diper_engine.js';
import { renderResults } from './results.js';

// Kept across visits so results survive navigating away and back.
const state = { sources: [], result: null, meta: null, settings: null };

function demoSources() {
  return simulateFiles().map(f => ({ file: new File([f.text], f.name, { type: 'text/csv' }), text: f.text, name: f.name, simulated: true }));
}

function sourceNote(src) {
  if (!src.length) return '<span class="mute">No data loaded yet.</span>';
  if (src[0].simulated) {
    return `<span class="sim-badge">SIMULATED</span> ${src.length} files × ${DEMO.tracks} persistent random walks (seed ${DEMO.seed}),
      x/y in µm, a constant speed of ${DEMO.minStep}-${DEMO.maxStep} µm per frame drawn for each cell. Not measured data.`;
  }
  return `Loaded ${src.length} file${src.length === 1 ? '' : 's'} (one condition each${isStatic() ? '' : '; one per sheet for Excel'}): `
    + src.map(s => `<b>${esc(s.name)}</b>${s.text ? ` (${s.text.split('\n').filter(Boolean).length - 1} rows)` : ''}`).join(', ');
}

const UNITS = ['min', 's', 'h', 'frames'];

route('analyse', async (args, query) => {
  const info = await api('/info');
  const d = info.defaults;
  const s = state.settings || { dt: d.time_interval, unit: 'min', n: d.max_intervals, sparse: d.sparse_n, coord: 'px', px: '', weighting: d.weighting };
  const engine = isStatic()
    ? 'On this static site the analysis runs <b>in your browser</b> with a JavaScript port of the Python package (tested to agree with it to about 1e-15). Nothing is uploaded.'
    : 'In the local app the analysis runs in <b>Python</b>: the files go to this computer\'s server, which calls the package\'s own functions.';
  app().innerHTML = `<p class="kicker">Analysis</p><h1>Analyse trajectories</h1>
    <p class="lead">All DiPer programs at once on one or more conditions: speed, directionality ratio, MSD, direction and
      velocity autocorrelation, and the tracks themselves.</p>
    <div class="callout a small">${engine}</div>
    <div class="panel"><h3 style="margin-top:0">1 · Data</h3>
      <label for="f-file">Track files, one per condition (${isStatic() ? 'CSV' : 'CSV or Excel'}; DiPer layout: frame, x, y in columns 4-6. <a href="#/format">Format</a>)</label>
      <input id="f-file" type="file" multiple accept="${isStatic() ? '.csv,text/csv' : '.csv,.xlsx,.xls'}">
      <button id="b-demo" class="alt" type="button">Use simulated demo data</button>
      <button id="b-demo-dl" class="alt" type="button">Download demo CSVs</button>
      <p id="src" class="small">${sourceNote(state.sources)}</p></div>
    <div class="panel"><h3 style="margin-top:0">2 · Units and settings</h3>
      <div class="row">
        <div><label for="s-dt">Time interval between frames</label><input id="s-dt" type="number" min="0" step="any" value="${s.dt}"></div>
        <div><label for="s-unit">Time unit</label><select id="s-unit">${UNITS.map(u => `<option ${u === s.unit ? 'selected' : ''}>${u}</option>`).join('')}</select></div>
      </div>
      <div class="row">
        <div><label for="s-coord">x / y coordinates are in</label><select id="s-coord"><option value="px" ${s.coord === 'px' ? 'selected' : ''}>pixels</option><option value="um" ${s.coord === 'um' ? 'selected' : ''}>micrometres (µm)</option></select></div>
        <div><label for="s-px">Pixel size (µm per pixel)</label><input id="s-px" type="number" min="0" step="any" value="${s.px}" placeholder="required for pixel coordinates"></div>
      </div>
      <div class="row">
        <div><label for="s-n">Autocorrel: number of intervals</label><input id="s-n" type="number" min="1" step="1" value="${s.n}"></div>
        <div><label for="s-sparse">Sparse_Data: keep 1 of N frames (1 = all)</label><input id="s-sparse" type="number" min="1" step="1" value="${s.sparse}"></div>
      </div>
      <div class="row">
        <div><label for="s-w">Autocorrel / Vel_Cor: average over cells</label><select id="s-w">
          <option value="displacements" ${s.weighting === 'displacements' ? 'selected' : ''}>weighted by each cell's displacements (paper, eq 5)</option>
          <option value="tracks" ${s.weighting === 'tracks' ? 'selected' : ''}>plain mean of the per-cell curves</option></select></div>
      </div>
      <p class="small mute">Coordinates are converted to µm before the analysis (speed in µm per time unit, MSD in µm²); the ratio,
        α and the autocorrelations do not depend on the scale. With Sparse_Data N &gt; 1 the time interval used is N × the one above.
        Defaults are those of the package's <code>run_*</code> functions (Δt ${d.time_interval}, ${d.max_intervals} intervals).</p>
      <button id="b-run" type="button">Analyse</button> <span id="status" class="small" role="status"></span></div>
    <div id="results"></div>`;

  const $ = id => document.getElementById(id);
  const coordSync = () => { $('s-px').disabled = $('s-coord').value === 'um'; };
  $('s-coord').onchange = coordSync; coordSync();
  const setSources = src => { state.sources = src; $('src').innerHTML = sourceNote(src); };
  const useDemo = () => {
    setSources(demoSources());
    $('s-coord').value = 'um'; coordSync();
    $('status').textContent = 'Simulated demo loaded (x/y in µm). Press Analyse.';
  };

  $('f-file').onchange = async e => {
    const list = [...e.target.files];
    if (!list.length) return;
    setSources(await Promise.all(list.map(async f => ({ file: f, text: /\.csv$/i.test(f.name) ? await f.text() : null, name: f.name, simulated: false }))));
  };
  $('b-demo').onclick = useDemo;
  $('b-demo-dl').onclick = () => simulateFiles().forEach(f => download(f.name, f.text, 'text/csv'));

  async function run() {
    const dt = +$('s-dt').value, n = +$('s-n').value, sparse = +$('s-sparse').value, coord = $('s-coord').value, px = +$('s-px').value, weighting = $('s-w').value;
    const status = $('status');
    const fail = msg => { status.innerHTML = `<span class="err">${msg}</span>`; };
    if (!state.sources.length) return fail('Load a file or the simulated demo first.');
    if (!(dt > 0)) return fail('The time interval must be a positive number.');
    if (!Number.isInteger(n) || n < 1) return fail('The number of intervals must be a whole number of at least 1.');
    if (!Number.isInteger(sparse) || sparse < 1) return fail('Sparse_Data N must be a whole number of at least 1.');
    if (coord === 'px' && !(px > 0)) return fail('Enter the pixel size in µm per pixel (or choose micrometre coordinates). It is never assumed.');
    state.settings = { dt, unit: $('s-unit').value, n, sparse, coord, px: coord === 'px' ? px : '', weighting };
    const fd = new FormData();
    for (const src of state.sources) fd.append('files', src.file, src.name);
    fd.append('time_interval', String(dt));
    fd.append('max_intervals', String(n));
    fd.append('sparse_n', String(sparse));
    fd.append('um_per_px', coord === 'px' ? String(px) : '');
    fd.append('weighting', weighting);
    $('b-run').disabled = true; status.textContent = 'Analysing...';
    const host = $('results');
    try {
      const res = await api('/analyze', { body: fd });
      let check = null;
      if (res.engine === 'python' && state.sources.every(x => x.text)) {
        try { check = maxDifference(analyseFiles(state.sources.map(x => ({ name: x.name, text: x.text })), dt, n, sparse, coord === 'px' ? px : null, weighting), res); } catch { check = null; }
      }
      state.result = res;
      state.meta = { simulated: state.sources[0].simulated, timeUnit: state.settings.unit, coord, check, maxSegments: info.max_segments };
      if (!host.isConnected) return;  // the user left the page meanwhile: results are kept for the next visit
      status.textContent = `Done (${res.engine === 'python' ? 'Python' : 'in-browser JS'}).`;
      renderResults(host, state.result, state.meta);
    } catch (e) {
      if (host.isConnected) fail(esc(e.message));
    } finally { if (host.isConnected) $('b-run').disabled = false; }
  }
  $('b-run').onclick = run;

  if ((args[0] === 'demo' || query.demo) && !state.result) {
    useDemo();
    await run();
  } else if (state.result) renderResults($('results'), state.result, state.meta);
});
