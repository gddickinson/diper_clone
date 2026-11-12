// SIMULATED demo trajectories (the repo has no example data: data/ is git-ignored), so this is not real data.
// Persistent random walk: the heading turns by a Gaussian angle N(0, sigma) radians every frame and each step has a
// fixed length (its own speed per cell, 1-2 um per frame). For this model the expected direction autocorrelation at a lag of m frames is exp(-m sigma^2 / 2)
// (also the normalized velocity autocorrelation, since the speed is constant); the results page overlays it.
// Output: one CSV per condition, in the DiPer column layout (Cell, Experiment, Condition, Frame, X, Y), x/y in um.
import { rng } from './core.js';

export const DEMO = {
  seed: 2014, tracks: 20, minLen: 30, maxLen: 80, minStep: 1, maxStep: 2, field: 400, umPerPx: null,
  conditions: [{ name: 'persistent_sigma_0.3', sigma: 0.3 }, { name: 'less_persistent_sigma_0.9', sigma: 0.9 }],
};

function normal(r) {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

// [{name, text}] : one simulated CSV file per condition.
export function simulateFiles(opts = {}) {
  const o = { ...DEMO, ...opts }, r = rng(o.seed);
  return o.conditions.map(c => {
    const lines = ['Cell,Experiment,Condition,Frame,X,Y'];
    for (let t = 1; t <= o.tracks; t++) {
      const len = o.minLen + Math.floor(r() * (o.maxLen - o.minLen + 1));
      const step = o.minStep + (o.maxStep - o.minStep) * r();
      let x = o.field * (0.2 + 0.6 * r()), y = o.field * (0.2 + 0.6 * r()), th = 2 * Math.PI * r();
      for (let f = 1; f <= len; f++) {
        lines.push(`cell${t},simulated,${c.name},${f},${x.toFixed(4)},${y.toFixed(4)}`);
        x += step * Math.cos(th); y += step * Math.sin(th);
        th += c.sigma * normal(r);
      }
    }
    return { name: `${c.name}.csv`, text: lines.join('\n') + '\n' };
  });
}

// Expected direction autocorrelation for the simulated model; steps are lags in frames.
export const theoryCurve = (sigma, steps) => steps.map(m => Math.exp(-m * sigma * sigma / 2));
