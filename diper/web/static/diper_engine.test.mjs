// node diper/web/static/diper_engine.test.mjs
// Asserts the JS port (diper_engine.js) reproduces the repo's Python analysis (diper/web/analysis.py, which calls
// load_data, split_trajectories, speed, dir_ratio, msd, autocorrel(_3d), vel_cor, plot_at_origin, sparse_data) on the
// inputs in tests/fixtures/diper_fixture.json (regenerate with: python scripts/make_diper_fixture.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { analyseFiles, maxDifference, splitTrajectories, velCorCurve, dirRatio, originArea } from './diper_engine.js';
import { toNum, conditionName } from './diper_csv.js';
import { simulateFiles, DEMO } from './diper_sim.js';

const TOL = 1e-12;
const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/diper_fixture.json', import.meta.url)));
let worst = 0, values = 0;
const count = o => (typeof o === 'number' ? 1 : o && typeof o === 'object' ? Object.values(o).reduce((s, v) => s + count(v), 0) : 0);
for (const c of fx.cases) {
  const got = analyseFiles(c.files, c.time_interval, c.max_intervals, c.sparse_n, c.um_per_px);
  const exp = c.expected;
  assert.equal(got.conditions.length, exp.conditions.length, c.name);
  got.conditions.forEach((g, i) => {
    const e = exp.conditions[i];
    assert.equal(g.name, e.name, `${c.name} name`);
    assert.equal(!!g.error, !!e.error, `${c.name}/${g.name} error: ${g.error} vs ${e.error}`);
    if (e.error) return;
    for (const k of ['n_rows', 'n_dropped', 'n_tracks', 'has_z', 'z_used']) assert.equal(g[k], e[k], `${c.name}/${g.name} ${k}`);
    for (const k of ['dir_ratio', 'msd', 'autocorr', 'vel_cor']) if (e[k]) assert.deepEqual(g[k].n, e[k].n, `${c.name}/${g.name} ${k}.n`);
  });
  const d = maxDifference(got, exp);
  values += count(exp.conditions);
  console.log(`${c.name}: ${got.conditions.length} condition(s), max relative |JS - Python| = ${d.toExponential(2)}`);
  assert.ok(d <= TOL, `${c.name}: difference ${d} > ${TOL}`);
  worst = Math.max(worst, d);
}

// Conventions of the Python code that the port must keep.
const tr = { frame: [1, 2, 3, 1, 2, 7, 3], x: [0, 1, 2, 5, 6, 7, 9], y: [0, 0, 0, 0, 0, 0, 0] };
assert.deepEqual(splitTrajectories(tr).map(t => t.x), [[0, 1, 2], [5, 6, 7]], 'non-increasing frame starts a track; 1-point tracks dropped');
const straight = { x: [0, 2, 4, 6, 8, 10, 12], y: [0, 1, 2, 3, 4, 5, 6] };
assert.ok(velCorCurve(straight, 0.5).v.every(v => Math.abs(v - 1) < 1e-12), 'Vel_Cor of constant velocity is 1 (eq 6 mean norm)');
assert.deepEqual(dirRatio({ x: [3, 4, 4], y: [0, 0, 1] }, 2).t, [0, 2, 4], 'Dir_Ratio time is elapsed time');
assert.deepEqual(originArea([{ x: [0, 42], y: [0, -7] }]), [-50, 50, -50, 50]);
assert.ok(Number.isNaN(toNum('0x10')) && toNum(' 2.5e1 ') === 25 && Number.isNaN(toNum('NA')));
assert.equal(conditionName('dir/My cells.CSV'), 'My cells');
assert.throws(() => analyseFiles([{ name: 'a.xlsx', text: '' }], 1, 5), /Excel/);
assert.throws(() => analyseFiles([{ name: 'a.csv', text: 'a,b,c,f,x,y\n' }], 0, 5), /positive/);

// Simulator: reproducible, in the DiPer layout, and analysable.
const s1 = simulateFiles(), s2 = simulateFiles();
assert.deepEqual(s1, s2);
assert.equal(s1.length, DEMO.conditions.length);
const r = analyseFiles(s1, 1, 10, 1, DEMO.umPerPx);
assert.ok(r.conditions.every(c => !c.error && c.n_tracks === DEMO.tracks && c.autocorr.avg[0] > 0));

console.log(`diper_engine: ${fx.cases.length} fixture cases, ${values} values, worst relative |JS - Python| = ${worst.toExponential(2)} (tol ${TOL}) ok`);
